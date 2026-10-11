# Last edited: 2026-10-11 12:00
"""Draw trade-off charts from results/bench.json."""

import sys
from pathlib import Path

import matplotlib.pyplot as plt

from mlxbench.results import load_records


def main() -> None:
    path = Path(sys.argv[1] if len(sys.argv) > 1 else "results/bench.json")
    recs = load_records(path)
    base = [r for r in recs if r.batch_size == 1]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4))
    for r in base:
        a1.scatter(r.peak_memory_gb, r.decode_tps)
        a1.annotate(
            f"{Path(r.model).name} ppl={r.perplexity and round(r.perplexity, 2)}",
            (r.peak_memory_gb, r.decode_tps),
        )
    a1.set(xlabel="peak memory (GB)", ylabel="decode tokens/s", title="Speed vs memory")
    for m in sorted({r.model for r in base}):
        pts = sorted((r.context_tokens, r.decode_tps) for r in base if r.model == m)
        a2.plot(*zip(*pts), marker="o", label=Path(m).name)
    a2.set(xlabel="context tokens", ylabel="decode tokens/s", xscale="log", title="Speed vs context")
    a2.legend()
    out = path.with_suffix(".png")
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    print(out)


if __name__ == "__main__":
    main()
