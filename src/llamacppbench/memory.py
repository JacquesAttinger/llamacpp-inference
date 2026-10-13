# Last edited: 2026-10-13 10:50
"""Peak memory helpers."""

import resource

GB = 1e9


def reset_peak() -> None:
    """Reset the peak-memory counter before a measured run."""
    # TODO(human): llama.cpp has no "reset peak memory" call, and ru_maxrss below can only go up.
    # Pick a way to get a fresh peak for each measured run, then make this function (and peak_gb) do it.
    #   Option A: measure in a fresh subprocess per run and read resource.getrusage(RUSAGE_CHILDREN).
    #   Option B: compute it from sizes (model file size + KV cache size). Predictable, but an estimate.
    # Whatever you choose, keep the two names reset_peak() and peak_gb(), or update timing.py and
    # batching.py to match. Write down in the README what the number really means.


def peak_gb() -> float:
    """Peak memory in GB since the last reset."""
    # Placeholder so the benchmark runs: the highest memory this whole process has ever held.
    # On macOS ru_maxrss is in bytes (on Linux it is in kilobytes).
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / GB
