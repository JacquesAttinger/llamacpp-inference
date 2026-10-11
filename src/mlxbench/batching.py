# Last edited: 2026-10-11 12:00
"""Batched generation with mlx-lm's batch_generate (verified in mlx-lm 0.32)."""

from mlx_lm.generate import batch_generate

from mlxbench.memory import reset_peak
from mlxbench.timing import TimingResult


def time_batch(
    model, tokenizer, prompt_tokens: list[int], batch_size: int, max_tokens: int = 128
) -> TimingResult:
    """Run batch_size copies of one prompt; report aggregate tokens per second."""
    reset_peak()  # Reset peak memory usage before running the generation
    stats = batch_generate(model, tokenizer, [prompt_tokens] * batch_size, max_tokens=max_tokens).stats
    # Approximation for TTFT, but this is technically just the prefill time.
    # It does not take into account system overheads.
    ttft = stats.prompt_tokens / stats.prompt_tps if stats.prompt_tps else 0.0
    return TimingResult(stats.prompt_tps, stats.generation_tps, ttft, stats.peak_memory)
