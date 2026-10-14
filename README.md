<!-- Last edited: 2026-10-14 11:22 -->
# llamacpp-bench

A small project for learning to benchmark LLM inference on Apple silicon.
It loads Llama 3.1 8B Instruct (GGUF files) with llama.cpp, through `llama-cpp-python`.
The benchmarking code is not here yet. It is being built from scratch.

## Setup

```bash
uv sync
uv run pre-commit install
```

The model files live in `models/` (16-bit, Q8_0, and Q4_K_M).
They are not in git. The script that made them is in the `v0-bench-reference` tag (`scripts/quantize.sh`).

## What is here

`src/llamacppbench/loading.py` has `load_model(path, n_ctx, logits_all)`.
It returns a `Llama` model and a `Tokenizer` with an `.encode(text)` method.

## Old version

The first full benchmark harness (timing, memory, perplexity, batching, plots) is kept for reference.
Read it with `git show v0-bench-reference:<path>` or `git checkout v0-bench-reference`.

See `docs/llama_cpp_conversion_plan.md` for the old design.
The first design (MLX) is in `docs/mlx_inference_scaffold_plan.md`.
