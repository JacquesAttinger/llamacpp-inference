# Last edited: 2026-10-13 10:55
"""Perplexity on fixed WikiText-2 chunks."""

import math


def perplexity_from_nll(total_nll: float, total_tokens: int) -> float:
    return math.exp(total_nll / total_tokens)


def chunk_tokens(tokens: list[int], chunk_len: int) -> list[list[int]]:
    """Split into full chunks of chunk_len; drop the short tail."""
    return [tokens[i : i + chunk_len] for i in range(0, len(tokens) - chunk_len + 1, chunk_len)]


def chunk_nll(model, chunk: list[int]) -> tuple[float, int]:
    """Summed negative log-likelihood and token count for one chunk.

    `model` must be loaded with logits_all=True and n_ctx >= len(chunk).
    """
    # TODO(human): score one chunk with numpy. The old MLX version fed chunk[:-1] to the model and
    # compared its output with chunk[1:] using cross_entropy(reduction="sum"). Do the same here:
    #   1. model.reset(), then model.eval(chunk[:-1])  (feeds the tokens, keeps all their scores)
    #   2. model.scores[:n] is a (n, vocab) float32 numpy array of raw scores, one row per position
    #   3. turn each row into log-probabilities (log-softmax) and pick the entry for the true next token
    #   4. return (sum of the negative log-probabilities, n)  where n = len(chunk) - 1
    # Watch out: exp() of raw scores overflows, so subtract each row's max before you exponentiate.
    raise NotImplementedError("chunk_nll is a TODO(human); run bench.py with --no-ppl until it is done")


def wikitext2_tokens(tokenizer, split: str = "test") -> list[int]:
    from datasets import load_dataset

    text = "\n\n".join(load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split=split)["text"])
    return tokenizer.encode(text)


def wikitext2_perplexity(model, tokenizer, chunk_len: int = 1024, max_chunks: int = 40) -> float:
    chunks = chunk_tokens(wikitext2_tokens(tokenizer), chunk_len)[:max_chunks]
    pairs = [chunk_nll(model, c) for c in chunks]
    return perplexity_from_nll(sum(p[0] for p in pairs), sum(p[1] for p in pairs))
