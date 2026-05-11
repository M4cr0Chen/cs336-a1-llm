import torch
import fire

from cs336_basics.config import ModelConfig
from cs336_basics.model import TransformerLM
from cs336_basics.generate import generate
from cs336_basics.tokenizer.tokenizer import load_tokenizer_from_dir


def main(
    prompt: str = "Once upon a time",
    checkpoint: str = "checkpoints/tiny_stories_transformer/best_model_step_10000.pt",
    model_config_json: str = "checkpoints/tiny_stories_transformer/model_config.json",
    dataset_dir: str = "datasets/tiny_stories",
    max_new_tokens: int = 256,
    temperature: float = 0.8,
    top_k: int = 50,
    top_p: float = 0.0,
):
    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"

    # Load model
    config = ModelConfig.from_json(model_config_json)
    model = TransformerLM(config).to(device)

    state = torch.load(checkpoint, map_location=device)
    model.load_state_dict(state["model_state_dict"])
    model.eval()

    # Load tokenizer
    tokenizer = load_tokenizer_from_dir(dataset_dir)

    # Generate
    result = generate(
        model=model,
        prompt=prompt,
        tokenizer=tokenizer,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_k=top_k,
        top_p=top_p,
    )

    print(result["all_text"])


if __name__ == "__main__":
    fire.Fire(main)
