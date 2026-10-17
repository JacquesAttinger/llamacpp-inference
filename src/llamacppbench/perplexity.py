# Last edited: 2026-10-16 08:15
"""Wikitext-2 perplexity with llama-cpp-python, scored the way llama.cpp's `llama-perplexity` tool does it.

Rules copied from llama.cpp (tools/perplexity/perplexity.cpp, the default path):
- The whole text becomes one token list. It is cut into back-to-back chunks of exactly n_ctx tokens.
  The leftover tail that does not fill a chunk is dropped.
- The first token of every chunk is replaced by the start token (BOS).
- Each chunk is run on its own. Only the second half of the chunk is scored,
  so every scored token has at least n_ctx / 2 tokens of context before it.
- Perplexity is exp(mean negative log-probability of the real next token).
"""

import io
import math
import urllib.request
import zipfile
from collections.abc import Callable
from pathlib import Path

import numpy as np

from llamacppbench.loading import Tokenizer

# The same file llama.cpp's own scripts/get-wikitext-2.sh downloads.
WIKITEXT_ZIP_URL = "https://huggingface.co/datasets/ggml-org/ci/resolve/main/wikitext-2-raw-v1.zip"
WIKITEXT_TEST_MEMBER = "wikitext-2-raw/wiki.test.raw"
DEFAULT_N_CTX = 512  # llama-perplexity's default chunk size


def ensure_wikitext_test(path: Path) -> Path:
    """Return `path`, downloading the Wikitext-2 raw test text there first if the file is missing."""
    if path.exists():
        return path
    print(f"Downloading Wikitext-2 test text to {path} ...", flush=True)
    with urllib.request.urlopen(WIKITEXT_ZIP_URL) as response:  # noqa: S310 - fixed https URL
        archive = zipfile.ZipFile(io.BytesIO(response.read()))
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(archive.read(WIKITEXT_TEST_MEMBER))
    return path


def token_logprobs(logits: np.ndarray, targets: np.ndarray) -> np.ndarray:
    """Log-probability of each target token. Row i of `logits` holds the scores for the token at targets[i].

    This is a log-softmax: subtract the row maximum first (so exp() cannot overflow), then subtract
    the log of the sum of exp. The sum is done in float64, like llama.cpp does.
    """
    shifted = logits - logits.max(axis=1, keepdims=True)
    log_total = np.log(np.exp(shifted).sum(axis=1, dtype=np.float64))
    chosen = shifted[np.arange(len(targets)), targets]
    return chosen - log_total


def score_chunk(model, chunk: list[int], n_ctx: int) -> np.ndarray:
    """Run one chunk through the model and return the log-probabilities of its scored tokens."""
    first = n_ctx // 2  # positions before this are context only
    model.reset()  # every chunk starts with an empty memory
    model.eval(chunk)
    # Row i of model.scores predicts chunk[i + 1], so rows first..n_ctx-2 predict chunk[first+1:].
    logits = model.scores[first : n_ctx - 1, :]
    targets = np.array(chunk[first + 1 : n_ctx])
    return token_logprobs(logits, targets)


def wikitext_perplexity(
    model,
    tokenizer: Tokenizer,
    text: str,
    n_ctx: int = DEFAULT_N_CTX,
    max_chunks: int | None = None,
    on_chunk: Callable[[int, int, float], None] | None = None,
) -> dict:
    """Perplexity of `text`. The model must be loaded with n_ctx=n_ctx and logits_all=True.

    on_chunk(done, total, running_perplexity) is called after each chunk, for progress output.
    Returns chunks, scored_tokens, mean_nll and perplexity.
    """
    tokens = tokenizer.encode(text)  # starts with the BOS token
    total = len(tokens) // n_ctx
    if max_chunks is not None:
        total = min(total, max_chunks)
    nll_sum = 0.0
    scored = 0
    for i in range(total):
        chunk = tokens[i * n_ctx : (i + 1) * n_ctx]
        chunk[0] = model.token_bos()
        logprobs = score_chunk(model, chunk, n_ctx)
        nll_sum -= float(logprobs.sum())
        scored += len(logprobs)
        if on_chunk is not None:
            on_chunk(i + 1, total, math.exp(nll_sum / scored))
    mean_nll = nll_sum / scored
    return {"chunks": total, "scored_tokens": scored, "mean_nll": mean_nll, "perplexity": math.exp(mean_nll)}
