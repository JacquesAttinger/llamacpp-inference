# Last edited: 2026-10-11 12:00
"""Load an MLX model and its tokenizer from a local path or Hugging Face repo."""

from mlx_lm import load

DEFAULT_MODEL = "meta-llama/Llama-3.1-8B-Instruct"


def load_model(path_or_repo: str = DEFAULT_MODEL):
    """Return (model, tokenizer) for a path or repo id."""
    model, tokenizer = load(path_or_repo)
    return model, tokenizer
