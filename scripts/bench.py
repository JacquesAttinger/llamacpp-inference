# Last edited: 2026-10-13 11:15
"""Run the benchmark grid and save JSON to results/."""

import argparse
from pathlib import Path

from llamacppbench.batching import time_batch
from llamacppbench.loading import DEFAULT_MODEL, load_model
from llamacppbench.perplexity import wikitext2_perplexity
from llamacppbench.results import BenchRecord, save_records
from llamacppbench.timing import time_generation

CONTEXTS = [128, 1024, 4096, 16384]  #
BATCHES = [1, 2, 4, 8]
BATCH_MAX_CONTEXT = 1024  # 8 x 16k KV cache would not fit in 24 GB
PPL_CHUNK = 1024  # tokens per perplexity chunk, which is also the n_ctx of the perplexity model
MAX_NEW_TOKENS = 128  # what timing.time_generation generates by default


def make_prompt(tokenizer, n: int) -> list[int]:
    """
    Make a prompt of n tokens, so we can see how input size changes the results.
    """
    ids = tokenizer.encode("The quick brown fox jumps over the lazy dog. ")
    return (ids * (n // len(ids) + 1))[:n]


def measure_perplexity(model_path: str) -> float:
    """Needs scores for every position, which is why it gets its own small-context model."""
    model, tok = load_model(model_path, n_ctx=PPL_CHUNK, logits_all=True)
    try:
        return wikitext2_perplexity(model, tok, chunk_len=PPL_CHUNK)
    finally:
        model.close()  # give the GPU memory back before the timing model loads


def run_model(model_path: str, contexts: list[int], batches: list[int], ppl: bool) -> list[BenchRecord]:
    perplexity = measure_perplexity(model_path) if ppl else None
    model, tok = load_model(model_path, n_ctx=max(contexts) + MAX_NEW_TOKENS)
    records = []

    for ctx in contexts:  # We loop over the different prompt sizes
        prompt = make_prompt(tok, ctx)

        for b in batches:  # We loop over the different batch sizes
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
    model.close()  # Cleans up memory that was used during the model run
    return records


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--models",
        nargs="+",
        default=[DEFAULT_MODEL, "models/llama-3.1-8b-q8_0.gguf", "models/llama-3.1-8b-q4_k_m.gguf"],
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
