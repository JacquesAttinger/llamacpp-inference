# Last edited: 2026-10-16 07:45
"""Measure how quantization changes decode speed (tokens per second) for Llama 3.1 8B.

For each model (16-bit, 8-bit, 4-bit): load it once, run 7 trials on the birthday prompt,
drop the first 2 as warm-up, and average decode tokens/s over the other 5.

Run from anywhere:  uv run python results/Experiments/quantization_decode_tps/run_experiment.py
Redraw the plot from the saved CSVs without running a model:  ... run_experiment.py --plot-only
"""

import argparse
import csv
import gc
import os
import statistics
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # draw to a file, no window
import matplotlib.pyplot as plt  # noqa: E402

EXPERIMENT_DIR = Path(__file__).resolve().parent
REPO_ROOT = EXPERIMENT_DIR.parents[2]
sys.path.insert(0, str(REPO_ROOT / "scripts"))  # so we can reuse run.py

from run import MIN_N_CTX, MODELS, PROMPT, generate_timed  # noqa: E402

from llamacppbench.loading import load_model  # noqa: E402

BITS_ORDER = [16, 8, 4]
WARMUP_TRIALS = 2
MEASURED_TRIALS = 5
TRIALS_CSV = EXPERIMENT_DIR / "trials.csv"
SUMMARY_CSV = EXPERIMENT_DIR / "summary.csv"
PLOT_PATH = EXPERIMENT_DIR / "decode_tps_vs_bits.png"
TRIAL_COLUMNS = ["bits", "trial", "phase", "generated_tokens", "prefill_s", "decode_s", "decode_tps"]
SUMMARY_COLUMNS = ["bits", "trials", "mean_decode_tps", "std_decode_tps"]


def run_model(bits: int) -> list[dict]:
    """Load one model, run all trials on it, and return one row per trial."""
    model, tokenizer = load_model(MODELS[bits], n_ctx=MIN_N_CTX)
    prompt_tokens = tokenizer.encode(PROMPT)
    rows = []
    for trial in range(1, WARMUP_TRIALS + MEASURED_TRIALS + 1):
        result = generate_timed(model, prompt_tokens)
        phase = "warmup" if trial <= WARMUP_TRIALS else "measured"
        print(f"{bits}-bit trial {trial} ({phase}): {result['decode_tps']:.2f} tokens/s", flush=True)
        rows.append(
            {
                "bits": bits,
                "trial": trial,
                "phase": phase,
                "generated_tokens": result["generated_tokens"],
                "prefill_s": round(result["prefill_s"], 4),
                "decode_s": round(result["decode_s"], 4),
                "decode_tps": round(result["decode_tps"], 2),
            }
        )
    model.close()
    del model
    gc.collect()  # give the memory back before the next model loads
    return rows


def write_csv(path: Path, columns: list[str], rows: list[dict]) -> None:
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def read_csv(path: Path) -> list[dict]:
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def measured_tps(trial_rows: list[dict], bits: int) -> list[float]:
    """Decode tokens/s of the measured (not warm-up) trials for one model."""
    return [float(r["decode_tps"]) for r in trial_rows if int(r["bits"]) == bits and r["phase"] == "measured"]


def summarize(trial_rows: list[dict]) -> list[dict]:
    """Mean and sample standard deviation of decode tokens/s over the measured trials of each model."""
    summary = []
    for bits in BITS_ORDER:
        values = measured_tps(trial_rows, bits)
        summary.append(
            {
                "bits": bits,
                "trials": len(values),
                "mean_decode_tps": round(statistics.mean(values), 2),
                "std_decode_tps": round(statistics.stdev(values), 2),
            }
        )
    return summary


def plot(trial_rows: list[dict], summary: list[dict]) -> None:
    """Mean decode tokens/s per model, with the individual measured trials as dots."""
    fig, ax = plt.subplots(figsize=(6, 4))
    bits = [int(s["bits"]) for s in summary]
    means = [float(s["mean_decode_tps"]) for s in summary]
    stds = [float(s["std_decode_tps"]) for s in summary]
    for b in bits:
        values = measured_tps(trial_rows, b)
        ax.scatter([b] * len(values), values, color="0.6", s=14, zorder=2)
    ax.errorbar(bits, means, yerr=stds, marker="o", color="tab:blue", capsize=4, zorder=3)
    for b, m in zip(bits, means, strict=True):
        ax.annotate(f"{m:.1f}", (b, m), textcoords="offset points", xytext=(0, 10), ha="center")
    ax.set_xscale("log", base=2)  # 4, 8, 16 are evenly spaced on a log2 axis
    ax.set_xticks(bits, [str(b) for b in bits])
    ax.minorticks_off()
    ax.set_xlabel("Bits per weight")
    ax.set_ylabel("Decode speed (tokens/s)")
    ax.set_title("Llama 3.1 8B: decode speed vs quantization")
    ax.set_ylim(0, max(means) * 1.25)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(PLOT_PATH, dpi=150)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--plot-only", action="store_true", help="redraw the plot from the saved CSVs")
    args = parser.parse_args()
    os.chdir(REPO_ROOT)  # the model paths in run.py are relative to the repo root

    if args.plot_only:
        trial_rows = read_csv(TRIALS_CSV)
    else:
        trial_rows = []
        for bits in BITS_ORDER:
            trial_rows += run_model(bits)
        write_csv(TRIALS_CSV, TRIAL_COLUMNS, trial_rows)
    summary = summarize(trial_rows)
    write_csv(SUMMARY_CSV, SUMMARY_COLUMNS, summary)
    plot(trial_rows, summary)
    for s in summary:
        print(f"{s['bits']}-bit: mean {s['mean_decode_tps']} tokens/s (std {s['std_decode_tps']})")
    print(f"Wrote {TRIALS_CSV.name}, {SUMMARY_CSV.name} and {PLOT_PATH.name}")


if __name__ == "__main__":
    main()
