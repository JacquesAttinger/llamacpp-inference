# Last edited: 2026-10-16 07:25
"""Load a model, tokenize one prompt, print the generated tokens, and log the timings to a CSV."""

import argparse
import csv
import time
from datetime import datetime
from pathlib import Path

from llamacppbench.loading import load_model

MODELS = {  # bits per weight -> model file; add a line here to support a new model
    4: "models/llama-3.1-8b-q4_k_m.gguf",
    8: "models/llama-3.1-8b-q8_0.gguf",
    16: "models/llama-3.1-8b-f16.gguf",
}
DEFAULT_BITS = 8
PROMPT = (
    "Come up with one idea for a happy birthday text message for my mom's 54th birthday. "
    "Give only one, not several."
)
MAX_TOKENS = 100  # safety stop in case the model never emits an end-of-text token
MIN_N_CTX = 2048  # context size when the prompt is short; longer prompts get a bigger one
CSV_PATH = Path("results/runs.csv")
CSV_COLUMNS = [
    "timestamp",
    "model",
    "temp",
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


def make_prompt(model, tokenizer, n_tokens: int) -> list[int]:
    """Exactly n_tokens ids: one start token, then PROMPT's tokens repeated and cut to fit."""
    body = tokenizer.encode(PROMPT, add_bos=False)  # no start token, so it appears only once below
    needed = n_tokens - 1  # one slot is taken by the start token
    repeated = body * (needed // len(body) + 1)
    return [model.token_bos()] + repeated[:needed]


def generate_timed(model, prompt_tokens: list[int], temp: float = 0.0, on_token=None) -> dict:
    """Generate up to MAX_TOKENS and time the prefill and the decode steps.

    temp 0.0 always takes the most likely token (greedy). A higher temp picks at random, so the text varies.
    """
    model.reset()  # forget the cached prompt, so every call pays for prefill again
    # generate() is lazy: each next() call asks the model for one more token.
    # The first next() reads the whole prompt (prefill) and returns token 1.
    # Each later next() is one decode step.
    # We time only the next() calls, so printing the text does not count.
    stream = model.generate(prompt_tokens, temp=temp)
    step_times = []  # seconds each next() call took
    tokens = []
    while len(tokens) < MAX_TOKENS:
        start = time.perf_counter()
        token = next(stream)
        step_times.append(time.perf_counter() - start)
        if token == model.token_eos():
            break
        tokens.append(token)
        if on_token is not None:
            on_token(token)

    # Token 1 came out of the prefill call, so the decode steps made tokens 2 to N.
    prefill_s = step_times[0]
    decode_s = sum(step_times[1 : len(tokens)])
    decode_tps = (len(tokens) - 1) / decode_s if decode_s > 0 else 0.0
    return {
        "generated_tokens": len(tokens),
        "prefill_s": prefill_s,
        "decode_s": decode_s,
        "decode_tps": decode_tps,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate text from a prompt and log the timings.")
    parser.add_argument(
        "--tokens",
        type=int,
        help="make the prompt exactly this many tokens by repeating the prompt text (default: use it once)",
    )
    parser.add_argument(
        "--bits",
        type=int,
        choices=sorted(MODELS),
        default=DEFAULT_BITS,
        help=f"which model to run, by bits per weight (default: {DEFAULT_BITS})",
    )
    parser.add_argument(
        "--temp",
        type=float,
        default=0.0,
        help="sampling temperature; 0 always takes the most likely token (default: 0)",
    )
    args = parser.parse_args()
    if args.tokens is not None and args.tokens < 1:
        parser.error("--tokens must be at least 1")
    if args.temp < 0:
        parser.error("--temp must not be negative")

    run_time = datetime.now().astimezone().isoformat(timespec="seconds")
    n_ctx = max(MIN_N_CTX, (args.tokens or 0) + MAX_TOKENS)  # the prompt and the new tokens must both fit
    model, tokenizer = load_model(MODELS[args.bits], n_ctx=n_ctx)
    if args.tokens is None:
        prompt_tokens = tokenizer.encode(PROMPT)
    else:
        prompt_tokens = make_prompt(model, tokenizer, args.tokens)
    print(f"Prompt text: {PROMPT}")
    print(f"Prompt tokens ({len(prompt_tokens)}): {prompt_tokens[:10]}...")
    print("Output: ", end="", flush=True)

    def show(token: int) -> None:
        print(model.detokenize([token]).decode(errors="replace"), end="", flush=True)

    result = generate_timed(model, prompt_tokens, temp=args.temp, on_token=show)
    print()
    model.close()

    append_row(
        {
            "timestamp": run_time,
            "model": f"{args.bits}-bit",
            "temp": args.temp,
            "prompt_tokens": len(prompt_tokens),
            "generated_tokens": result["generated_tokens"],
            "prefill_s": round(result["prefill_s"], 4),
            "decode_s": round(result["decode_s"], 4),
            "decode_tps": round(result["decode_tps"], 2),
        }
    )
    print(
        f"prefill {result['prefill_s']:.3f}s, decode {result['decode_s']:.3f}s, "
        f"{result['decode_tps']:.1f} tokens/s -> {CSV_PATH}"
    )


if __name__ == "__main__":
    main()
