# Last edited: 2026-10-16 08:20
"""Measure how quantization changes perplexity on the Wikitext-2 test text for Llama 3.1 8B.

For each model (16-bit, 8-bit, 4-bit): load it once and score the whole test text in 512-token chunks,
the way llama.cpp's `llama-perplexity` does. Perplexity does not depend on timing or sampling,
so each model is run once. Lower perplexity is better.

Run from anywhere:  uv run python results/Experiments/quantization_perplexity/run_experiment.py
Try it on a few chunks of one model first:  ... run_experiment.py --bits 4 --max-chunks 2   (writes nothing)
Redraw the plot from the saved CSVs without running a model:  ... run_experiment.py --plot-only
"""

import argparse
import csv
import gc
import os
import sys
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # draw to a file, no window
import matplotlib.pyplot as plt  # noqa: E402

EXPERIMENT_DIR = Path(__file__).resolve().parent
REPO_ROOT = EXPERIMENT_DIR.parents[2]
sys.path.insert(0, str(REPO_ROOT / "scripts"))  # so we can reuse run.py

from run import MODELS  # noqa: E402

from llamacppbench.loading import load_model  # noqa: E402
from llamacppbench.perplexity import DEFAULT_N_CTX, ensure_wikitext_test, wikitext_perplexity  # noqa: E402

BITS_ORDER = [16, 8, 4]
DATA_PATH = REPO_ROOT / "data" / "wiki.test.raw"
TRIALS_CSV = EXPERIMENT_DIR / "trials.csv"
SUMMARY_CSV = EXPERIMENT_DIR / "summary.csv"
PLOT_PATH = EXPERIMENT_DIR / "perplexity_vs_bits.png"
TRIAL_COLUMNS = ["bits", "trial", "phase", "chunks", "scored_tokens", "mean_nll", "perplexity", "wall_s"]
SUMMARY_COLUMNS = ["bits", "runs", "perplexity", "change_vs_16bit_pct"]
PROGRESS_EVERY = 50  # chunks between progress lines


def run_model(bits: int, text: str, max_chunks: int | None) -> dict:
    """Load one model, score the text, and return one result row."""
    model, tokenizer = load_model(MODELS[bits], n_ctx=DEFAULT_N_CTX, logits_all=True)
    start = time.perf_counter()

    def progress(done: int, total: int, running: float) -> None:
        if done % PROGRESS_EVERY == 0 or done == total:
            print(f"{bits}-bit: chunk {done}/{total}, running perplexity {running:.4f}", flush=True)

    result = wikitext_perplexity(model, tokenizer, text, DEFAULT_N_CTX, max_chunks, progress)
    wall_s = time.perf_counter() - start
    model.close()
    del model
    gc.collect()  # give the memory back before the next model loads
    print(f"{bits}-bit: perplexity {result['perplexity']:.4f} in {wall_s / 60:.1f} min", flush=True)
    return {
        "bits": bits,
        "trial": 1,
        "phase": "measured",
        "chunks": result["chunks"],
        "scored_tokens": result["scored_tokens"],
        "mean_nll": round(result["mean_nll"], 6),
        "perplexity": round(result["perplexity"], 4),
        "wall_s": round(wall_s, 1),
    }


def write_csv(path: Path, columns: list[str], rows: list[dict]) -> None:
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def read_csv(path: Path) -> list[dict]:
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def summarize(trial_rows: list[dict]) -> list[dict]:
    """Perplexity per model, and how much higher it is than the 16-bit model's."""
    by_bits = {int(r["bits"]): float(r["perplexity"]) for r in trial_rows if r["phase"] == "measured"}
    baseline = by_bits[16]
    return [
        {
            "bits": bits,
            "runs": sum(1 for r in trial_rows if int(r["bits"]) == bits and r["phase"] == "measured"),
            "perplexity": round(by_bits[bits], 4),
            "change_vs_16bit_pct": round((by_bits[bits] / baseline - 1) * 100, 2),
        }
        for bits in BITS_ORDER
    ]


def plot(summary: list[dict]) -> None:
    """Perplexity per model. One run each, so there are no error bars."""
    fig, ax = plt.subplots(figsize=(6, 4))
    bits = [int(s["bits"]) for s in summary]
    values = [float(s["perplexity"]) for s in summary]
    ax.plot(bits, values, marker="o", color="tab:blue", zorder=3)
    for b, v in zip(bits, values, strict=True):
        ax.annotate(f"{v:.3f}", (b, v), textcoords="offset points", xytext=(0, 10), ha="center")
    ax.set_xscale("log", base=2)  # 4, 8, 16 are evenly spaced on a log2 axis
    ax.set_xticks(bits, [str(b) for b in bits])
    ax.minorticks_off()
    ax.invert_xaxis()  # 16 on the left, 4 on the right
    ax.set_xlabel("Bits per weight")
    ax.set_ylabel("Perplexity on Wikitext-2 test (lower is better)")
    ax.set_title("Llama 3.1 8B: perplexity vs quantization")
    low, high = min(values), max(values)
    pad = max((high - low) * 0.5, 0.05)  # keep the labels inside the plot
    ax.set_ylim(low - pad, high + pad)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(PLOT_PATH, dpi=150)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--plot-only", action="store_true", help="redraw the plot from the saved CSVs")
    parser.add_argument(
        "--max-chunks",
        type=int,
        help="score only this many chunks per model, as a smoke test (writes nothing)",
    )
    parser.add_argument(
        "--bits", type=int, choices=BITS_ORDER, help="run only this model, as a smoke test (writes nothing)"
    )
    args = parser.parse_args()
    os.chdir(REPO_ROOT)  # the model paths in run.py are relative to the repo root

    if args.plot_only:
        trial_rows = read_csv(TRIALS_CSV)
    else:
        text = ensure_wikitext_test(DATA_PATH).read_text(encoding="utf-8")
        to_run = [args.bits] if args.bits else BITS_ORDER
        trial_rows = [run_model(bits, text, args.max_chunks) for bits in to_run]
        if args.max_chunks is not None or args.bits is not None:
            return  # a partial run must not overwrite the real results
        write_csv(TRIALS_CSV, TRIAL_COLUMNS, trial_rows)
    summary = summarize(trial_rows)
    write_csv(SUMMARY_CSV, SUMMARY_COLUMNS, summary)
    plot(summary)
    for s in summary:
        print(f"{s['bits']}-bit: perplexity {s['perplexity']} ({s['change_vs_16bit_pct']:+}% vs 16-bit)")
    print(f"Wrote {TRIALS_CSV.name}, {SUMMARY_CSV.name} and {PLOT_PATH.name}")


if __name__ == "__main__":
    main()
