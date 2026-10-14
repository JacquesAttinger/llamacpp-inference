# Last edited: 2026-10-14 12:00
"""Perplexity on fixed WikiText-2 chunks."""

import math

import numpy as np


def perplexity_from_nll(total_nll: float, total_tokens: int) -> float:
    return math.exp(total_nll / total_tokens)


def chunk_tokens(tokens: list[int], chunk_len: int) -> list[list[int]]:
    """Split into full chunks of chunk_len; drop the short tail."""
    return [tokens[i : i + chunk_len] for i in range(0, len(tokens) - chunk_len + 1, chunk_len)]


def chunk_nll(model, chunk: list[int]) -> tuple[float, int]:
    """Summed negative log-likelihood and token count for one chunk.

    `model` must be loaded with logits_all=True and n_ctx >= len(chunk).
    """
    n = len(chunk) - 1
    model.reset()  # otherwise llama-cpp-python may reuse the previous chunk's KV cache
    model.eval(chunk[:-1])
    return nll_from_scores(model.scores[:n], chunk[1:]), n


def nll_from_scores(scores: np.ndarray, targets: list[int]) -> float:
    """Summed negative log-likelihood of `targets`, given one row of raw scores per target."""
    scores = scores.astype(np.float64)  # float64 keeps the sum over ~1000 rows accurate
    # log-softmax: subtract the row max first so exp() cannot overflow.
    shifted = scores - scores.max(axis=1, keepdims=True)
    log_probs = shifted - np.log(np.exp(shifted).sum(axis=1, keepdims=True))
    picked = np.take_along_axis(log_probs, np.asarray(targets)[:, None], axis=1)
    return float(-picked.sum())


def wikitext2_tokens(tokenizer, split: str = "test") -> list[int]:
    from datasets import load_dataset

    text = "\n\n".join(load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split=split)["text"])
    return tokenizer.encode(text)


def wikitext2_perplexity(model, tokenizer, chunk_len: int = 1024, max_chunks: int = 40) -> float:
    chunks = chunk_tokens(wikitext2_tokens(tokenizer), chunk_len)[:max_chunks]
    pairs = [chunk_nll(model, c) for c in chunks]
    return perplexity_from_nll(sum(p[0] for p in pairs), sum(p[1] for p in pairs))
