# Last edited: 2026-10-11 12:30
"""Load an MLX model and its tokenizer from a local path or Hugging Face repo."""

from mlx_lm import load

# Local HF-cache snapshot (the folder that holds config.json and the safetensors shards).
DEFAULT_MODEL = (
    "/Users/jacquesattinger/.cache/huggingface/hub/"
    "models--meta-llama--Llama-3.1-8B-Instruct/snapshots/0e9e39f249a16976918f6564b8830bc894c89659"
)


def load_model(path_or_repo: str = DEFAULT_MODEL):
    """Return (model, tokenizer) for a path or repo id."""
    model, tokenizer = load(path_or_repo)
    return model, tokenizer
