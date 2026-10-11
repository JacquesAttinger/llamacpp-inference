# Last edited: 2026-10-11 12:00
"""Time prefill (time to first token) and decode separately."""

import statistics
from dataclasses import dataclass

from mlx_lm import stream_generate

from mlxbench.memory import peak_gb, reset_peak


@dataclass
class TimingResult:
    # MLX-lm gets this value by counting the number of tokens in the prompt
    # and dividing it by the time it took to ingest them.
    prefill_tps: float
    decode_tps: float  # tokens per second generated during decode step
    time_to_first_token_s: float  # time of first token generated - time of ingestion of the prompt
    peak_memory_gb: float  # Peak memory used during inference run


def summarize(runs: list[TimingResult]) -> TimingResult:
    """Median of each field across repeated runs."""
    return TimingResult(
        *(statistics.median(getattr(r, f) for r in runs) for f in TimingResult.__annotations__)
    )


def time_once(model, tokenizer, prompt_tokens: list[int], max_tokens: int) -> TimingResult:
    """One generation; prefill and decode speeds come from mlx-lm's own counters."""
    reset_peak()
    last = None
    for last in stream_generate(model, tokenizer, prompt_tokens, max_tokens=max_tokens):
        pass
    ttft = len(prompt_tokens) / last.prompt_tps
    return TimingResult(last.prompt_tps, last.generation_tps, ttft, peak_gb())


def time_generation(
    model, tokenizer, prompt_tokens: list[int], max_tokens: int = 128, warmup: int = 1, repeats: int = 3
) -> TimingResult:
    """Warm up, then take the median of `repeats` runs."""
    for _ in range(warmup):
        # Runs time_once `warmup` times to warm up the model before measuring performance.
        time_once(model, tokenizer, prompt_tokens, 8)
    return summarize(
        [time_once(model, tokenizer, prompt_tokens, max_tokens) for _ in range(repeats)]
    )  # Summarize does multiple runs and takes the median of their statistics
