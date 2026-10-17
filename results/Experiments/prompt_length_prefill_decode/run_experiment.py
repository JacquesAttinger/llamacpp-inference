# Last edited: 2026-10-17 11:30
"""Measure how prompt length changes prefill time and decode speed for Llama 3.1 8B (4-bit).

Load the 4-bit model once and run 2 warm-up trials on a 1000-token prompt (dropped).
Then, for prompt lengths 100, 200, ... 1000 tokens (in that order), run 3 trials each
and average prefill seconds and decode tokens/s over the 3.

Run from anywhere:  uv run python results/Experiments/prompt_length_prefill_decode/run_experiment.py
Redraw the plots from the saved CSVs without running a model:  ... run_experiment.py --plot-only
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

from run import MAX_TOKENS, MIN_N_CTX, MODELS, generate_timed, make_prompt  # noqa: E402

from llamacppbench.loading import load_model  # noqa: E402

BITS = 4
PROMPT_LENGTHS = list(range(100, 1001, 100))
WARMUP_TRIALS = 2
WARMUP_LENGTH = max(PROMPT_LENGTHS)
MEASURED_TRIALS = 3
N_CTX = max(MIN_N_CTX, max(PROMPT_LENGTHS) + MAX_TOKENS)  # same context size for every length
TRIALS_CSV = EXPERIMENT_DIR / "trials.csv"
SUMMARY_CSV = EXPERIMENT_DIR / "summary.csv"
PREFILL_PLOT = EXPERIMENT_DIR / "prefill_s_vs_prompt_tokens.png"
DECODE_PLOT = EXPERIMENT_DIR / "decode_tps_vs_prompt_tokens.png"
METRICS = ["prefill_s", "prefill_tps", "decode_s", "decode_tps"]
TRIAL_COLUMNS = ["prompt_tokens", "trial", "phase", "generated_tokens", *METRICS]
SUMMARY_COLUMNS = ["prompt_tokens", "trials"] + [f"{p}_{m}" for m in METRICS for p in ("mean", "std")]


def run_trial(model, tokenizer, prompt_tokens: int, trial: int, phase: str) -> dict:
    """Time one generation on a prompt of exactly prompt_tokens tokens and return one CSV row."""
    prompt = make_prompt(model, tokenizer, prompt_tokens)
    result = generate_timed(model, prompt)
    row = {
        "prompt_tokens": prompt_tokens,
        "trial": trial,
        "phase": phase,
        "generated_tokens": result["generated_tokens"],
        "prefill_s": round(result["prefill_s"], 4),
        "prefill_tps": round(prompt_tokens / result["prefill_s"], 2),
        "decode_s": round(result["decode_s"], 4),
        "decode_tps": round(result["decode_tps"], 2),
    }
    print(
        f"{prompt_tokens:>4} tokens, trial {trial} ({phase}): prefill {row['prefill_s']:.3f}s, "
        f"decode {row['decode_tps']:.2f} tokens/s, wrote {row['generated_tokens']} tokens",
        flush=True,
    )
    return row


def run_all() -> list[dict]:
    """Load the model once, run the warm-up, then every prompt length in order."""
    model, tokenizer = load_model(MODELS[BITS], n_ctx=N_CTX)
    rows = []
    for trial in range(1, WARMUP_TRIALS + 1):
        rows.append(run_trial(model, tokenizer, WARMUP_LENGTH, trial, "warmup"))
    for length in PROMPT_LENGTHS:
        for trial in range(1, MEASURED_TRIALS + 1):
            rows.append(run_trial(model, tokenizer, length, trial, "measured"))
    model.close()
    del model
    gc.collect()  # give the memory back
    return rows


def write_csv(path: Path, columns: list[str], rows: list[dict]) -> None:
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def read_csv(path: Path) -> list[dict]:
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def measured(trial_rows: list[dict], length: int, metric: str) -> list[float]:
    """The metric's values from the measured (not warm-up) trials at one prompt length."""
    return [
        float(r[metric]) for r in trial_rows if int(r["prompt_tokens"]) == length and r["phase"] == "measured"
    ]


def summarize(trial_rows: list[dict]) -> list[dict]:
    """Mean and sample standard deviation of every metric over the measured trials at each length."""
    summary = []
    for length in PROMPT_LENGTHS:
        row = {"prompt_tokens": length, "trials": len(measured(trial_rows, length, "prefill_s"))}
        for metric in METRICS:
            values = measured(trial_rows, length, metric)
            row[f"mean_{metric}"] = round(statistics.mean(values), 4)
            row[f"std_{metric}"] = round(statistics.stdev(values), 4)
        summary.append(row)
    return summary


def plot(trial_rows: list[dict], summary: list[dict], metric: str, ylabel: str, title: str, path: Path):
    """Mean of one metric against prompt length, with the individual trials as grey dots."""
    fig, ax = plt.subplots(figsize=(6.5, 4))
    lengths = [int(s["prompt_tokens"]) for s in summary]
    means = [float(s[f"mean_{metric}"]) for s in summary]
    stds = [float(s[f"std_{metric}"]) for s in summary]
    for length in lengths:
        values = measured(trial_rows, length, metric)
        ax.scatter([length] * len(values), values, color="0.6", s=14, zorder=2)
    ax.errorbar(lengths, means, yerr=stds, marker="o", color="tab:blue", capsize=4, zorder=3)
    # the label sits above whichever is higher: the top error bar or the highest dot
    highs = [
        max(m + s, *measured(trial_rows, x, metric)) for x, m, s in zip(lengths, means, stds, strict=True)
    ]
    top = max(highs)
    for x, m, high in zip(lengths, means, highs, strict=True):
        ax.annotate(f"{m:.2f}", (x, high), textcoords="offset points", xytext=(0, 6), ha="center", fontsize=8)
    ax.set_ylim(0, top * 1.2)
    ax.set_xticks(lengths)  # lengths are evenly spaced, so a linear axis is right
    ax.set_xlabel("Prompt length (tokens)")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--plot-only", action="store_true", help="redraw the plots from the saved CSVs")
    args = parser.parse_args()
    os.chdir(REPO_ROOT)  # the model paths in run.py are relative to the repo root

    if args.plot_only:
        trial_rows = read_csv(TRIALS_CSV)
    else:
        trial_rows = run_all()
        write_csv(TRIALS_CSV, TRIAL_COLUMNS, trial_rows)
    summary = summarize(trial_rows)
    write_csv(SUMMARY_CSV, SUMMARY_COLUMNS, summary)
    plot(
        trial_rows, summary, "prefill_s", "Prefill time (s)",
        "Llama 3.1 8B (4-bit): prefill time vs prompt length", PREFILL_PLOT,
    )  # fmt: skip
    plot(
        trial_rows, summary, "decode_tps", "Decode speed (tokens/s)",
        "Llama 3.1 8B (4-bit): decode speed vs prompt length", DECODE_PLOT,
    )  # fmt: skip
    for s in summary:
        print(
            f"{s['prompt_tokens']:>4} tokens: prefill {s['mean_prefill_s']}s "
            f"(std {s['std_prefill_s']}), decode {s['mean_decode_tps']} tokens/s (std {s['std_decode_tps']})"
        )
    print(f"Wrote {TRIALS_CSV.name}, {SUMMARY_CSV.name}, {PREFILL_PLOT.name} and {DECODE_PLOT.name}")


if __name__ == "__main__":
    main()
