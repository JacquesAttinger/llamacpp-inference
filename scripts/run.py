# Last edited: 2026-10-16 07:45
"""Load a model, tokenize one prompt, print the generated tokens, and log the timings to a CSV."""

import argparse
import csv
import random
import time
from datetime import datetime
from pathlib import Path

from llamacppbench.loading import load_model
from llamacppbench.sampling import LOCAL_PATH, NAMES, load_sampling

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
MAX_SEED = 2**32 - 2  # 2**32 - 1 is llama.cpp's "random" marker, so it is not a usable seed
MIN_N_CTX = 2048  # context size when the prompt is short; longer prompts get a bigger one
CSV_PATH = Path("results/runs.csv")
CSV_COLUMNS = [
    "timestamp",
    "model",
    "temp",
    "seed",
    *NAMES,  # top_k, top_p, min_p, typical_p
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


def generate_timed(
    model, prompt_tokens: list[int], temp: float = 0.0, sampling: dict | None = None, on_token=None
) -> dict:
    """Generate up to MAX_TOKENS and time the prefill and the decode steps.

    temp 0.0 always takes the most likely token (greedy). A higher temp picks at random, so the text varies.
    sampling holds top_k, top_p, min_p and typical_p. Leave it out to use llama-cpp-python's own defaults.
    """
    model.reset()  # forget the cached prompt, so every call pays for prefill again
    # generate() is lazy: each next() call asks the model for one more token.
    # The first next() reads the whole prompt (prefill) and returns token 1.
    # Each later next() is one decode step.
    # We time only the next() calls, so printing the text does not count.
    stream = model.generate(prompt_tokens, temp=temp, **(sampling or {}))
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


def parse_args() -> argparse.Namespace:
    """Read and check the command-line flags."""
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
    parser.add_argument(
        "--seed",
        type=int,
        help=(
            "seed for the random picks when temp > 0; the same seed and settings give the same text "
            "(default: a random one, which is logged)"
        ),
    )
    args = parser.parse_args()
    if args.tokens is not None and args.tokens < 1:
        parser.error("--tokens must be at least 1")
    if args.temp < 0:
        parser.error("--temp must not be negative")
    if args.seed is not None and not 0 <= args.seed <= MAX_SEED:
        parser.error(f"--seed must be between 0 and {MAX_SEED}")
    return args


def main() -> None:
    args = parse_args()
    try:
        sampling, overrides = load_sampling()
    except ValueError as error:
        raise SystemExit(f"error: {error}")
    if overrides:
        changes = ", ".join(f"{name}={value}" for name, value in overrides.items())
        print(f"Using overrides from {LOCAL_PATH.name}: {changes}")
    # Pick the random seed ourselves, so the log can replay the run.
    seed = args.seed if args.seed is not None else random.randint(0, MAX_SEED)

    run_time = datetime.now().astimezone().isoformat(timespec="seconds")
    n_ctx = max(MIN_N_CTX, (args.tokens or 0) + MAX_TOKENS)  # the prompt and the new tokens must both fit
    model, tokenizer = load_model(MODELS[args.bits], n_ctx=n_ctx, seed=seed)
    if args.tokens is None:
        prompt_tokens = tokenizer.encode(PROMPT)
    else:
        prompt_tokens = make_prompt(model, tokenizer, args.tokens)
    print(f"Prompt text: {PROMPT}")
    print(f"Prompt tokens ({len(prompt_tokens)}): {prompt_tokens[:10]}...")
    print("\nOutput:")  # the reply sits between quotation marks, so it is easy to tell from the other lines
    print('"', end="", flush=True)

    def show(token: int) -> None:
        print(model.detokenize([token]).decode(errors="replace"), end="", flush=True)

    result = generate_timed(model, prompt_tokens, temp=args.temp, sampling=sampling, on_token=show)
    print('"\n')
    model.close()

    append_row(
        {
            "timestamp": run_time,
            "model": f"{args.bits}-bit",
            "temp": args.temp,
            "seed": seed,
            **sampling,
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
