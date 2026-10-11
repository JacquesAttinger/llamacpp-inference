# Last edited: 2026-10-11 12:00
"""Perplexity on fixed WikiText-2 chunks."""

import math

import mlx.core as mx
import mlx.nn as nn


def perplexity_from_nll(total_nll: float, total_tokens: int) -> float:
    return math.exp(total_nll / total_tokens)


def chunk_tokens(tokens: list[int], chunk_len: int) -> list[list[int]]:
    """Split into full chunks of chunk_len; drop the short tail."""
    return [tokens[i : i + chunk_len] for i in range(0, len(tokens) - chunk_len + 1, chunk_len)]


def chunk_nll(model, chunk: list[int]) -> tuple[float, int]:
    """Summed negative log-likelihood and token count for one chunk."""
    x = mx.array(chunk)[None]
    logits = model(x[:, :-1])
    loss = nn.losses.cross_entropy(logits, x[:, 1:], reduction="sum")
    mx.eval(loss)
    return loss.item(), x.shape[1] - 1


def wikitext2_tokens(tokenizer, split: str = "test") -> list[int]:
    from datasets import load_dataset

    text = "\n\n".join(load_dataset("wikitext", "wikitext-2-raw-v1", split=split)["text"])
    return tokenizer.encode(text)


def wikitext2_perplexity(model, tokenizer, chunk_len: int = 1024, max_chunks: int = 40) -> float:
    chunks = chunk_tokens(wikitext2_tokens(tokenizer), chunk_len)[:max_chunks]
    pairs = [chunk_nll(model, c) for c in chunks]
    return perplexity_from_nll(sum(p[0] for p in pairs), sum(p[1] for p in pairs))
