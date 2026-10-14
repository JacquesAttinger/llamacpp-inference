# Last edited: 2026-10-14 11:47
"""Load a GGUF model with llama.cpp (through llama-cpp-python)."""

from llama_cpp import Llama

# Written by scripts/quantize.sh. Relative to the repo root.
DEFAULT_MODEL = "models/llama-3.1-8b-f16.gguf"


class Tokenizer:
    """Gives the model's built-in llama.cpp tokenizer the `.encode(text)` shape the rest of the code uses."""

    def __init__(self, llm: Llama):
        self._llm = llm

    def encode(self, text: str, add_bos: bool = True) -> list[int]:
        # add_bos=True matches what the Hugging Face tokenizer did for Llama 3.
        return self._llm.tokenize(text.encode(), add_bos=add_bos)


def load_model(
    path: str = DEFAULT_MODEL, n_ctx: int = 2048, logits_all: bool = False
) -> tuple[Llama, Tokenizer]:
    """Return (model, tokenizer) for a GGUF file.

    n_ctx is the KV cache size in tokens. A bigger value uses more memory, so size it for the run.
    logits_all keeps the scores for every position (perplexity needs them) and costs n_ctx * vocab * 4 bytes.
    """
    model = Llama(
        model_path=path,
        n_gpu_layers=-1,  # -1 puts every layer on the GPU (Metal)
        n_ctx=n_ctx,
        n_batch=2048,  # tokens fed to the model per prefill step (llama.cpp's own default)
        n_ubatch=512,  # the physical chunk inside one prefill step (llama.cpp's own default)
        flash_attn=True,  # smaller scratch memory for long contexts
        # False copies the weights into the process's own memory. With mmap, macOS may keep only some weight
        # pages counted as ours, so the peak-memory readings jump around by several GB between runs.
        use_mmap=False,
        logits_all=logits_all,
        verbose=False,
    )
    return model, Tokenizer(model)
