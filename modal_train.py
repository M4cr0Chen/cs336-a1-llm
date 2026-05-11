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


@app.local_entrypoint()
def main(
    train_config_json: str | None = None,
    model_config_json: str | None = None,
):
    train.remote(
        train_config_json=train_config_json,
        model_config_json=model_config_json,
    )
