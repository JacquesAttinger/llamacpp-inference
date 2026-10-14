# Last edited: 2026-10-14 12:05
"""Load a model, tokenize one prompt, print the generated tokens, and log the timings to a CSV."""

import csv
import time
from datetime import datetime
from pathlib import Path

from llamacppbench.loading import load_model

MODEL_PATH = "models/llama-3.1-8b-q8_0.gguf"
# MODEL_PATH = "models/llama-3.1-8b-f16.gguf"
MODEL_LABELS = {
    "models/llama-3.1-8b-f16.gguf": "16-bit",
    "models/llama-3.1-8b-q8_0.gguf": "8-bit",
}
PROMPT = (
    "Come up with one idea for a happy birthday text message for my mom's 54th birthday. "
    "Give only one, not several."
)
MAX_TOKENS = 100  # safety stop in case the model never emits an end-of-text token
CSV_PATH = Path("results/runs.csv")
CSV_COLUMNS = [
    "timestamp",
    "model",
    "prompt_tokens",
    "generated_tokens",
    "prefill_s",
    "decode_s",
    "decode_tps",
]


def append_row(row: dict) -> None:
    """Add one row to the CSV, writing the header first if the file is new or empty."""
    CSV_PATH.parent.mkdir(parents=True, exist_ok=True)
    write_header = not CSV_PATH.exists() or CSV_PATH.stat().st_size == 0
    with CSV_PATH.open("a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
        if write_header:
            writer.writeheader()
        writer.writerow(row)


def main() -> None:
    run_time = datetime.now().astimezone().isoformat(timespec="seconds")
    model, tokenizer = load_model(MODEL_PATH)
    prompt_tokens = tokenizer.encode(PROMPT)
    print(f"Prompt: {PROMPT}")
    print(f"Prompt tokens ({len(prompt_tokens)}): {prompt_tokens}")
    print("Output: ", end="", flush=True)

    # generate() is lazy: each next() call asks the model for one more token.
    # The first next() reads the whole prompt (prefill) and returns token 1.
    # Each later next() is one decode step.
    # We time only the next() calls, so printing the text does not count.
    stream = model.generate(prompt_tokens, temp=0.0)
    step_times = []  # seconds each next() call took
    tokens = []
    while len(tokens) < MAX_TOKENS:
        start = time.perf_counter()
        token = next(stream)
        step_times.append(time.perf_counter() - start)
        if token == model.token_eos():
            break
        tokens.append(token)
        print(model.detokenize([token]).decode(errors="replace"), end="", flush=True)
    print()
    model.close()

    # Token 1 came out of the prefill call, so the decode steps made tokens 2 to N.
    prefill_s = step_times[0]
    decode_s = sum(step_times[1 : len(tokens)])
    decode_tps = (len(tokens) - 1) / decode_s if decode_s > 0 else 0.0
    append_row(
        {
            "timestamp": run_time,
            "model": MODEL_LABELS[MODEL_PATH],
            "prompt_tokens": len(prompt_tokens),
            "generated_tokens": len(tokens),
            "prefill_s": round(prefill_s, 4),
            "decode_s": round(decode_s, 4),
            "decode_tps": round(decode_tps, 2),
        }
    )
    print(f"prefill {prefill_s:.3f}s, decode {decode_s:.3f}s, {decode_tps:.1f} tokens/s -> {CSV_PATH}")


if __name__ == "__main__":
    main()
