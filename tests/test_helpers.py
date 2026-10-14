# Last edited: 2026-10-11 12:00
import math

import numpy as np

from llamacppbench.perplexity import chunk_tokens, nll_from_scores, perplexity_from_nll
from llamacppbench.results import BenchRecord, load_records, save_records
from llamacppbench.timing import TimingResult, summarize


def test_perplexity_uniform():
    assert math.isclose(perplexity_from_nll(math.log(4) * 10, 10), 4.0)


def test_nll_from_scores_uniform_and_large_scores():
    # Equal scores over 4 words: each target has probability 1/4. Huge scores must not overflow.
    scores = np.full((3, 4), 1000.0, dtype=np.float32)
    assert math.isclose(nll_from_scores(scores, [0, 1, 3]), 3 * math.log(4))


def test_chunk_tokens_drops_tail():
    assert chunk_tokens(list(range(10)), 4) == [[0, 1, 2, 3], [4, 5, 6, 7]]


def test_summarize_median():
    runs = [TimingResult(i, i, i, i) for i in (1, 5, 3)]
    assert summarize(runs).decode_tps == 3


def test_results_roundtrip(tmp_path):
    rec = BenchRecord("m", 128, 1, 1.0, 2.0, 0.1, 3.0, 5.5)
    save_records([rec], tmp_path / "r.json")
    assert load_records(tmp_path / "r.json") == [rec]
