<!-- Last edited: 2026-10-16 07:20 -->
# llamacpp-bench

A small project for learning to benchmark LLM inference on Apple silicon.
It loads Llama 3.1 8B Instruct (GGUF files) with llama.cpp, through `llama-cpp-python`.
The benchmarking code is not here yet. It is being built from scratch.

## Setup

```bash
uv sync
uv run pre-commit install
```

The model files live in `models/` (16-bit, Q8_0 and Q4_K_M).
They are not in git. The script that made them is in the `v0-bench-reference` tag (`scripts/quantize.sh`).

## Run a model from the command line

Run this from the repo root, because the model and CSV paths are relative:

```bash
uv run python scripts/run.py
```

The script loads a model, feeds it one prompt, and prints the reply as the model writes it.
It stops at the end-of-text token or after 100 new tokens, whichever comes first.
Then it prints the timings and adds one row to `results/runs.csv`.

To make the prompt longer, pass `--tokens`.
The script repeats the prompt text until it is exactly that many tokens long.
This is useful to see how the prefill time grows with the prompt size.

```bash
uv run python scripts/run.py --tokens 1000
```

To pick the model, pass `--bits` with `4`, `8` or `16`:

```bash
uv run python scripts/run.py --bits 4
```

| `--bits` | File | Notes |
| --- | --- | --- |
| `4` | `models/llama-3.1-8b-q4_k_m.gguf` | |
| `8` | `models/llama-3.1-8b-q8_0.gguf` | the default |
| `16` | `models/llama-3.1-8b-f16.gguf` | |

The `model` column in the CSV shows `4-bit`, `8-bit` or `16-bit`.
The table above is the `MODELS` dict at the top of `scripts/run.py`.
To add another model, add a line to that dict.
The prompt text is the `PROMPT` constant in the same file.

Each run adds a row with these columns:

- `prefill_s`: seconds the model spent reading the prompt, before it made the first token.
- `decode_s`: seconds it spent writing all the later tokens.
- `decode_tps`: tokens written per second during decode.

## What is here

`src/llamacppbench/loading.py` has `load_model(path, n_ctx, logits_all)`.
It returns a `Llama` model and a `Tokenizer` with an `.encode(text)` method.

`scripts/run.py` is the command-line script described above.

## Old version

The first full benchmark harness (timing, memory, perplexity, batching, plots) is kept for reference.
Read it with `git show v0-bench-reference:<path>` or `git checkout v0-bench-reference`.

See `docs/llama_cpp_conversion_plan.md` for the old design.
The first design (MLX) is in `docs/mlx_inference_scaffold_plan.md`.
