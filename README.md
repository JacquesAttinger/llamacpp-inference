<!-- Last edited: 2026-10-11 12:00 -->
# mlx-inference

Benchmarks Llama 3.1 8B Instruct on Apple silicon with MLX.
It compares 16-bit, 8-bit, and 4-bit weights on speed, memory, and perplexity.

## Setup

```bash
uv sync
uv run pre-commit install
scripts/quantize.sh          # makes models/llama-3.1-8b-q8 and -q4
```

## Run

```bash
uv run python scripts/bench.py --no-ppl --models meta-llama/Llama-3.1-8B-Instruct --contexts 128 --batches 1
uv run python scripts/bench.py   # full grid
uv run python scripts/plot.py    # results/bench.json -> results/bench.png
```

See `docs/mlx_inference_scaffold_plan.md` for the design.
