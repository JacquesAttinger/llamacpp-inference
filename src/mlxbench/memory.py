# Last edited: 2026-10-11 12:00
"""Peak GPU memory helpers."""

import mlx.core as mx

GB = 1e9


def reset_peak() -> None:
    """Reset the peak-memory counter before a measured run."""
    mx.reset_peak_memory()


def peak_gb() -> float:
    """Peak memory in GB since the last reset."""
    return mx.get_peak_memory() / GB
