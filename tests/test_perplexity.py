# Last edited: 2026-10-16 08:15
"""Tests for the log-probability maths behind the perplexity score."""

import math

import numpy as np

from llamacppbench.perplexity import token_logprobs


def test_equal_scores_give_one_over_vocab_size():
    logits = np.zeros((3, 8), dtype=np.float32)
    logprobs = token_logprobs(logits, np.array([0, 3, 7]))
    assert np.allclose(logprobs, -math.log(8))


def test_the_likely_token_gets_a_higher_logprob():
    logits = np.array([[5.0, 0.0, 0.0]], dtype=np.float32)
    assert token_logprobs(logits, np.array([0]))[0] > token_logprobs(logits, np.array([1]))[0]


def test_probabilities_of_all_tokens_sum_to_one():
    logits = np.array([[1.0, 2.0, 3.0, -1.0]], dtype=np.float32)
    total = sum(math.exp(token_logprobs(logits, np.array([t]))[0]) for t in range(4))
    assert math.isclose(total, 1.0, rel_tol=1e-6)


def test_huge_scores_do_not_overflow():
    logits = np.array([[1000.0, 999.0]], dtype=np.float32)
    assert np.isfinite(token_logprobs(logits, np.array([0]))).all()
