# Last edited: 2026-10-18 11:10
"""Measure how batching (1, 3, 5 or 7 prompts at once) changes throughput, prefill speed and decode speed.

Loads the 8-bit Llama 3.1 8B once. For each batch size it runs 1 warm-up trial (dropped) and 3 measured
trials, with every prompt the default birthday prompt writing exactly 100 greedy tokens. It saves the mean
and sample standard deviation of each speed.

Run from anywhere:  uv run python results/Experiments/batching_throughput/run_experiment.py
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

from batched_generate import BatchedModel  # noqa: E402
from run import DEFAULT_BITS, MODELS, PROMPT  # noqa: E402

BATCH_SIZES = [1, 3, 5, 7]
WARMUP_TRIALS = 1  # per batch size
MEASURED_TRIALS = 3
N_NEW = 100  # tokens written per prompt: token 1 comes from prefill, tokens 2 to 100 from decode steps
TRIALS_CSV = EXPERIMENT_DIR / "trials.csv"
SUMMARY_CSV = EXPERIMENT_DIR / "summary.csv"
PLOT_PATH = EXPERIMENT_DIR / "metrics_vs_batch_size.png"
METRICS = ["throughput_tps", "prefill_tps", "decode_tps_total", "decode_tps_per_stream"]
TRIAL_COLUMNS = ["n_prompts", "trial", "phase", "prompt_tokens", "prefill_s", "decode_s", *METRICS]
SUMMARY_COLUMNS = ["n_prompts", "trials", *[f"{p}_{m}" for m in METRICS for p in ("mean", "std")]]


def trial_row(n: int, trial: int, prompt_tokens: int, result: dict) -> dict:
    """One CSV row: the raw times plus the four speeds worked out from them."""
    prefill_s, decode_s = result["prefill_s"], result["decode_s"]
    decode_total = n * (N_NEW - 1) / decode_s  # all prompts' tokens 2..100 over the decode time
    return {
        "n_prompts": n,
        "trial": trial,
        "phase": "warmup" if trial <= WARMUP_TRIALS else "measured",
        "prompt_tokens": prompt_tokens,
        "prefill_s": round(prefill_s, 4),
        "decode_s": round(decode_s, 4),
        "throughput_tps": round(n * N_NEW / (prefill_s + decode_s), 2),  # generated tokens, whole job
        "prefill_tps": round(n * prompt_tokens / prefill_s, 2),  # prompt tokens read per second
        "decode_tps_total": round(decode_total, 2),
        "decode_tps_per_stream": round(decode_total / n, 2),
    }


def run_all() -> list[dict]:
    """Load the model once and run every batch size in order."""
    model = BatchedModel(MODELS[DEFAULT_BITS], n_seq_max=max(BATCH_SIZES))
    prompt = model.tokenize(PROMPT)
    rows = []
    for n in BATCH_SIZES:
        for trial in range(1, WARMUP_TRIALS + MEASURED_TRIALS + 1):
            row = trial_row(n, trial, len(prompt), model.generate(prompt, n, N_NEW))
            rows.append(row)
            print(
                f"{n} prompts trial {trial} ({row['phase']}): throughput {row['throughput_tps']} tokens/s, "
                f"prefill {row['prefill_tps']} tokens/s, decode {row['decode_tps_total']} tokens/s",
                flush=True,
            )
    model.close()
    del model
    gc.collect()
    return rows


def write_csv(path: Path, columns: list[str], rows: list[dict]) -> None:
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def read_csv(path: Path) -> list[dict]:
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def measured(trial_rows: list[dict], n: int, metric: str) -> list[float]:
    """The metric's value in each measured (not warm-up) trial at batch size n."""
    return [float(r[metric]) for r in trial_rows if int(r["n_prompts"]) == n and r["phase"] == "measured"]


def summarize(trial_rows: list[dict]) -> list[dict]:
    """Mean and sample standard deviation of every metric over the measured trials of each batch size."""
    summary = []
    for n in BATCH_SIZES:
        row = {"n_prompts": n, "trials": len(measured(trial_rows, n, METRICS[0]))}
        for metric in METRICS:
            values = measured(trial_rows, n, metric)
            row[f"mean_{metric}"] = round(statistics.mean(values), 2)
            row[f"std_{metric}"] = round(statistics.stdev(values), 2)
        summary.append(row)
    return summary


def draw_line(
    ax, trial_rows: list[dict], summary: list[dict], metric: str, color: str, label: str, skip_first=False
) -> None:
    """One metric's mean (with std error bars and labels) and its trials as grey dots."""
    means = [float(s[f"mean_{metric}"]) for s in summary]
    stds = [float(s[f"std_{metric}"]) for s in summary]
    for n in BATCH_SIZES:
        values = measured(trial_rows, n, metric)
        ax.scatter([n] * len(values), values, color="0.6", s=14, zorder=2)
    ax.errorbar(BATCH_SIZES, means, yerr=stds, marker="o", color=color, capsize=4, zorder=3, label=label)
    for i, (n, m) in enumerate(zip(BATCH_SIZES, means, strict=True)):
        if skip_first and i == 0:
            continue  # same point as another line's first label
        rising = i + 1 < len(means) and means[i + 1] > m
        # A line that rises out of the point would cross a label above it, so the label goes below it.
        offset, align = ((6, -16), "left") if rising else ((0, 16), "center")
        ax.annotate(f"{m:.1f}", (n, m), textcoords="offset points", xytext=offset, ha=align, fontsize=10)


def plot(trial_rows: list[dict], summary: list[dict]) -> None:
    """Three panels: throughput, prefill speed, and decode speed (whole batch and per prompt)."""
    plt.rcParams.update({"font.size": 12})  # the plot is printed small in the book, so keep text large
    fig, axes = plt.subplots(1, 3, figsize=(12, 4.4))
    panels = [
        (axes[0], "Throughput", [("throughput_tps", "tab:blue", None)]),
        (axes[1], "Prefill speed", [("prefill_tps", "tab:green", None)]),
        (
            axes[2],
            "Decode speed",
            [
                ("decode_tps_total", "tab:red", "whole batch"),
                ("decode_tps_per_stream", "tab:orange", "per prompt"),
            ],
        ),
    ]
    for ax, title, lines in panels:
        for k, (metric, color, label) in enumerate(lines):
            draw_line(ax, trial_rows, summary, metric, color, label, skip_first=k > 0)
        top = max(float(s[f"mean_{lines[0][0]}"]) for s in summary)
        ax.set_ylim(0, top * 1.2)
        ax.set_xticks(BATCH_SIZES)
        ax.margins(x=0.12)  # room so the labels at 1 and 7 stay inside the axes
        ax.set_xlabel("Prompts in the batch")
        ax.set_ylabel("Tokens per second")
        ax.set_title(title)
        ax.grid(axis="y", alpha=0.3)
    axes[2].legend(loc="upper left")
    fig.suptitle("Llama 3.1 8B (8-bit): effect of batching")
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
        trial_rows = run_all()
        write_csv(TRIALS_CSV, TRIAL_COLUMNS, trial_rows)
    summary = summarize(trial_rows)
    write_csv(SUMMARY_CSV, SUMMARY_COLUMNS, summary)
    plot(trial_rows, summary)
    for s in summary:
        print(
            f"{s['n_prompts']} prompts: throughput {s['mean_throughput_tps']}, "
            f"prefill {s['mean_prefill_tps']}, decode {s['mean_decode_tps_total']} "
            f"({s['mean_decode_tps_per_stream']} per prompt) tokens/s"
        )
    print(f"Wrote {TRIALS_CSV.name}, {SUMMARY_CSV.name} and {PLOT_PATH.name}")


if __name__ == "__main__":
    main()
