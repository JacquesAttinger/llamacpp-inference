# Last edited: 2026-10-11 12:00
"""Run the benchmark grid and save JSON to results/."""

import argparse
from pathlib import Path

from mlxbench.batching import time_batch
from mlxbench.loading import DEFAULT_MODEL, load_model
from mlxbench.perplexity import wikitext2_perplexity
from mlxbench.results import BenchRecord, save_records
from mlxbench.timing import time_generation

CONTEXTS = [128, 1024, 4096, 16384]
BATCHES = [1, 2, 4, 8]
BATCH_MAX_CONTEXT = 1024  # 8 x 16k KV cache would not fit in 24 GB


def make_prompt(tokenizer, n: int) -> list[int]:
    ids = tokenizer.encode("The quick brown fox jumps over the lazy dog. ")
    return (ids * (n // len(ids) + 1))[:n]


def run_model(model_path: str, contexts: list[int], batches: list[int], ppl: bool) -> list[BenchRecord]:
    model, tok = load_model(model_path)
    perplexity = wikitext2_perplexity(model, tok) if ppl else None
    records = []
    for ctx in contexts:
        prompt = make_prompt(tok, ctx)
        for b in batches:
            if b > 1 and ctx > BATCH_MAX_CONTEXT:
                continue
            t = time_generation(model, tok, prompt) if b == 1 else time_batch(model, tok, prompt, b)
            records.append(
                BenchRecord(
                    model_path,
                    ctx,
                    b,
                    t.prefill_tps,
                    t.decode_tps,
                    t.time_to_first_token_s,
                    t.peak_memory_gb,
                    perplexity,
                )
            )
            print(records[-1])
    return records


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--models", nargs="+", default=[DEFAULT_MODEL, "models/llama-3.1-8b-q8", "models/llama-3.1-8b-q4"]
    )
    p.add_argument("--contexts", nargs="+", type=int, default=CONTEXTS)
    p.add_argument("--batches", nargs="+", type=int, default=BATCHES)
    p.add_argument("--no-ppl", action="store_true")
    p.add_argument("--out", type=Path, default=Path("results/bench.json"))
    a = p.parse_args()
    records = [r for m in a.models for r in run_model(m, a.contexts, a.batches, not a.no_ppl)]
    save_records(records, a.out)


if __name__ == "__main__":
    main()
