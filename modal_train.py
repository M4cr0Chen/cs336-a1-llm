import modal

app = modal.App("cs336-llm-training")

# Persistent volume for datasets and checkpoints
volume = modal.Volume.from_name("cs336-data", create_if_missing=True)

# Build the container image with your project installed
image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install("uv")
    .add_local_dir("cs336_basics", remote_path="/root/project/cs336_basics", copy=True)
    .add_local_file("pyproject.toml", remote_path="/root/project/pyproject.toml", copy=True)
    .add_local_file("train.py", remote_path="/root/project/train.py", copy=True)
    .add_local_file("README.md", remote_path="/root/project/README.md", copy=True)
    .run_commands("cd /root/project && uv pip install --system -e .")
)

VOLUME_PATH = "/data"


@app.function(
    image=image,
    volumes={VOLUME_PATH: volume},
    timeout=3600,
)
def upload_dataset():
    """Upload local dataset files to the Modal volume.
    Run with: modal run modal_train.py::upload_dataset
    """
    import shutil
    from pathlib import Path

    dest = Path(VOLUME_PATH) / "tiny_stories"
    dest.mkdir(parents=True, exist_ok=True)

    # Copy from local mount (we use local_mount for the upload step)
    local_dir = Path("/root/project/datasets/tiny_stories")
    if local_dir.exists():
        for f in local_dir.iterdir():
            shutil.copy2(f, dest / f.name)
            print(f"Copied {f.name} ({(dest / f.name).stat().st_size / 1e6:.1f} MB)")
    else:
        print("No local dataset found. Upload manually or use the CLI approach below.")

    volume.commit()
    print("Dataset uploaded to volume.")


@app.function(
    image=image,
    gpu="H100",  # change to "H100", "A10G", "T4", etc. as needed
    volumes={VOLUME_PATH: volume},
    secrets=[modal.Secret.from_name("wandb")],
    timeout=7200,
)
def train(
    train_config_json: str | None = None,
    model_config_json: str | None = None,
):
    """Run training on a Modal GPU.
    Run with: modal run modal_train.py::train
    """
    import os
    import subprocess

    os.chdir("/root/project")

    # Symlink volume data into the expected paths
    # Volume has: /data/tiny_stories/{train.bin, eval.bin, vocab.json, ...}
    # Code expects: datasets/tiny_stories/... and checkpoints/...
    os.symlink(VOLUME_PATH, "/root/project/datasets")

    checkpoint_vol = f"{VOLUME_PATH}/checkpoints"
    os.makedirs(checkpoint_vol, exist_ok=True)
    os.symlink(checkpoint_vol, "/root/project/checkpoints")

    # Build the command
    cmd = ["python", "train.py"]
    if train_config_json:
        cmd += ["--train_config_json", train_config_json]
    if model_config_json:
        cmd += ["--model_config_json", model_config_json]

    print(f"Running: {' '.join(cmd)}")
    subprocess.run(cmd, check=True)

    # Persist checkpoints and any outputs to volume
    volume.commit()
    print("Training complete. Checkpoints saved to volume.")


@app.function(
    image=image,
    gpu="H100",
    volumes={VOLUME_PATH: volume},
    timeout=3600,
)
def benchmark():
    """Run all benchmarks on H100.
    Run with: modal run modal_train.py::benchmark
    """
    import os
    import time

    import numpy as np
    import torch

    os.chdir("/root/project")
    os.symlink(VOLUME_PATH, "/root/project/datasets")

    from cs336_basics.config import ModelConfig, TrainingConfig
    from cs336_basics.data import BatchState, data_loading_sequential
    from cs336_basics.generate import generate
    from cs336_basics.loss import cross_entropy
    from cs336_basics.model import TransformerLM
    from cs336_basics.optimizer import AdamW, cosine_annealing_lr, gradient_clip
    from cs336_basics.tokenizer.tokenizer import load_tokenizer_from_dir
    from cs336_basics.train_engine import eval_model
    from cs336_basics.utils import get_ctx

    device = "cuda"
    dataset_dir = "datasets/tiny_stories"
    checkpoint_dir = f"{VOLUME_PATH}/checkpoints/tiny_stories_transformer"
    checkpoint_path = None
    for f in sorted(os.listdir(checkpoint_dir)):
        if f.startswith("best_model") and f.endswith(".pt"):
            checkpoint_path = os.path.join(checkpoint_dir, f)
    if checkpoint_path is None:
        raise FileNotFoundError(f"No checkpoint found in {checkpoint_dir}")
    print(f"Using checkpoint: {checkpoint_path}")

    model_config = ModelConfig.from_json(os.path.join(checkpoint_dir, "model_config.json"))
    train_config = TrainingConfig()
    train_config.device = device

    # ── 1. Tokenizer Throughput ──────────────────────────────────
    print("\n" + "=" * 60)
    print("1. TOKENIZER THROUGHPUT")
    print("=" * 60)

    tokenizer = load_tokenizer_from_dir(dataset_dir)
    # Read eval data, decode to text for encoding benchmark
    raw_bytes = open(os.path.join(dataset_dir, "tiny_stories", "eval.bin"), "rb").read()
    # Use a chunk of text for benchmarking — decode the merges/vocab isn't useful,
    # so let's generate text by decoding token IDs from the eval set
    eval_data = np.memmap(
        train_config.eval_data_path, dtype=np.uint16, mode="r"
    )
    sample_ids = eval_data[:100_000].tolist()
    sample_text = tokenizer.decode(sample_ids)
    text_bytes = len(sample_text.encode("utf-8"))

    # Warmup
    tokenizer.encode(sample_text[:1000])

    num_runs = 5
    total_tokens = 0
    total_time = 0.0
    for _ in range(num_runs):
        t0 = time.perf_counter()
        tokens = tokenizer.encode(sample_text)
        t1 = time.perf_counter()
        total_tokens += len(tokens)
        total_time += t1 - t0

    tok_per_sec = total_tokens / total_time
    mb_per_sec = (text_bytes * num_runs / 1e6) / total_time
    print(f"  Text size: {text_bytes / 1e6:.2f} MB ({len(sample_text)} chars)")
    print(f"  Throughput: {tok_per_sec:,.0f} tokens/sec")
    print(f"  Throughput: {mb_per_sec:.1f} MB/s")

    # ── 2. Validation Perplexity ─────────────────────────────────
    print("\n" + "=" * 60)
    print("2. VALIDATION PERPLEXITY")
    print("=" * 60)

    model = TransformerLM(model_config).to(device)
    state = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(state["model_state_dict"])

    eval_loss, eval_ppl = eval_model(model, train_config)
    print(f"  Eval Loss: {eval_loss.item():.4f}")
    print(f"  Eval Perplexity: {eval_ppl.item():.4f}")

    # ── 3. Inference Latency ─────────────────────────────────────
    print("\n" + "=" * 60)
    print("3. INFERENCE LATENCY")
    print("=" * 60)

    model.eval()
    prompt = "Once upon a time"
    max_new_tokens = 128

    # Warmup
    generate(model=model, prompt=prompt, tokenizer=tokenizer,
             max_new_tokens=32, top_k=50, temperature=0.8)
    torch.cuda.synchronize()

    num_runs = 5
    latencies = []
    tokens_generated = []
    for _ in range(num_runs):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        result = generate(model=model, prompt=prompt, tokenizer=tokenizer,
                          max_new_tokens=max_new_tokens, top_k=50, temperature=0.8)
        torch.cuda.synchronize()
        t1 = time.perf_counter()
        n_tokens = len(result["generated_ids"])
        latencies.append((t1 - t0) / n_tokens * 1000)  # ms/token
        tokens_generated.append(n_tokens)

    avg_ms = sum(latencies) / len(latencies)
    avg_tokens = sum(tokens_generated) / len(tokens_generated)
    print(f"  Avg tokens generated: {avg_tokens:.0f}")
    print(f"  Latency: {avg_ms:.2f} ms/token")
    print(f"  Throughput: {1000 / avg_ms:.1f} tokens/sec")

    # ── 4. Training Throughput & MFU ─────────────────────────────
    print("\n" + "=" * 60)
    print("4. TRAINING THROUGHPUT & MFU")
    print("=" * 60)

    # Fresh model for training benchmark
    model = TransformerLM(model_config).to(device)
    model.train()
    num_params = sum(p.numel() for p in model.parameters())
    print(f"  Model parameters: {num_params:,}")

    optimizer = AdamW(model.parameters(), lr=train_config.min_lr,
                      betas=train_config.betas, weight_decay=train_config.weight_decay)

    train_data = np.memmap(train_config.train_data_path, dtype=np.uint16, mode="r")
    x = torch.from_numpy(train_data.copy())
    ctx = get_ctx(True, device)
    batch_state = BatchState(pos=0)
    tokens_per_step = train_config.batch_size * model_config.max_seq_len

    # Warmup 5 steps
    for _ in range(5):
        inputs, targets = data_loading_sequential(
            x=x, batch_size=train_config.batch_size,
            context_length=model_config.max_seq_len, device=device, state=batch_state)
        with ctx:
            logits, aux = model(inputs)
            loss = cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        gradient_clip(model.parameters(), max_l2_norm=train_config.max_grad_norm)
        optimizer.step()
    torch.cuda.synchronize()

    # Timed steps
    num_steps = 50
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    start.record()
    for step in range(num_steps):
        inputs, targets = data_loading_sequential(
            x=x, batch_size=train_config.batch_size,
            context_length=model_config.max_seq_len, device=device, state=batch_state)
        with ctx:
            logits, aux = model(inputs)
            loss = cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        gradient_clip(model.parameters(), max_l2_norm=train_config.max_grad_norm)
        lr = cosine_annealing_lr(step, train_config.max_lr, train_config.min_lr,
                                 train_config.warmup_steps, train_config.num_steps - train_config.warmup_steps)
        for pg in optimizer.param_groups:
            pg["lr"] = lr
        optimizer.step()
    end.record()
    torch.cuda.synchronize()

    elapsed_ms = start.elapsed_time(end)
    elapsed_s = elapsed_ms / 1000
    avg_step_ms = elapsed_ms / num_steps
    total_tokens = num_steps * tokens_per_step
    tokens_per_sec = total_tokens / elapsed_s

    # MFU: 6 * N flops per token (fwd + bwd), H100 SXM BF16 peak = 989.5 TFLOPS
    flops_per_token = 6 * num_params
    actual_tflops = (flops_per_token * tokens_per_sec) / 1e12
    h100_peak_tflops = 989.5
    mfu = actual_tflops / h100_peak_tflops * 100

    print(f"  Batch size: {train_config.batch_size}, Seq len: {model_config.max_seq_len}")
    print(f"  Tokens/step: {tokens_per_step:,}")
    print(f"  Avg step time: {avg_step_ms:.2f} ms")
    print(f"  Training throughput: {tokens_per_sec:,.0f} tokens/sec")
    print(f"  Actual TFLOPS: {actual_tflops:.2f}")
    print(f"  H100 peak BF16: {h100_peak_tflops} TFLOPS")
    print(f"  MFU: {mfu:.2f}%")

    # ── Summary ──────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("SUMMARY — Resume Numbers")
    print("=" * 60)
    print(f"  Tokenizer:  {tok_per_sec:,.0f} tokens/sec ({mb_per_sec:.1f} MB/s)")
    print(f"  Perplexity: {eval_ppl.item():.2f}")
    print(f"  Inference:  {avg_ms:.2f} ms/token")
    print(f"  Training:   {tokens_per_sec:,.0f} tokens/sec, MFU {mfu:.2f}%")


@app.local_entrypoint()
def main(
    train_config_json: str | None = None,
    model_config_json: str | None = None,
):
    train.remote(
        train_config_json=train_config_json,
        model_config_json=model_config_json,
    )
