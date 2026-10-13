# Last edited: 2026-10-13 10:45
"""Time prefill (time to first token) and decode separately."""

import statistics
from dataclasses import dataclass

import llama_cpp

from llamacppbench.memory import peak_gb, reset_peak


@dataclass
class TimingResult:
    # llama.cpp counts the prompt tokens it processed and the milliseconds that took,
    # and we divide one by the other.
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
    """One generation; prefill and decode speeds come from llama.cpp's own timers."""
    model.reset()  # forget the old KV cache, or llama-cpp-python reuses it and skips the prefill
    reset_peak()
    llama_cpp.llama_perf_context_reset(model.ctx)
    for _, _ in zip(range(max_tokens), model.generate(prompt_tokens, temp=0.0)):
        pass  # temp=0.0 is greedy, so every run does the same work
    perf = llama_cpp.llama_perf_context(model.ctx)
    prefill_tps = perf.n_p_eval / (perf.t_p_eval_ms / 1000)
    decode_tps = perf.n_eval / (perf.t_eval_ms / 1000)
    ttft = len(prompt_tokens) / prefill_tps
    return TimingResult(prefill_tps, decode_tps, ttft, peak_gb())


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
