<!-- Last edited: 2026-10-13 11:25 -->
> **Historical.** This is the original MLX design.
> The project now runs on llama.cpp. See `llama_cpp_conversion_plan.md`.

# Plan: scaffold `~/code/mlx-inference`

Plan date: 2026-10-11.

## TLDR

We make a new Python project that tests how fast Llama 3.1 8B runs on your Mac.
It makes smaller copies of the model (8-bit and 4-bit).
It measures speed, memory, and quality for each copy.
Then it draws charts that show the trade-off.

## Context

You want to learn AI inference by doing real optimization work.
Phase 1 runs on your Apple M5 Mac with MLX.
Phase 2 (later, optional) runs on a rented NVIDIA GPU with vLLM.
The scaffold is the empty project that holds all Phase 1 work.

## Facts checked

- `~/code/mlx-inference` does not exist yet.
- The Mac reports 24 GB of memory (not 32 GB).
  16-bit weights take about 16 GB, so context length and batch size are limited.
- `uv` 0.12.5 and `git` are installed.
  System Python is 3.14.6, which is too new for some wheels.
- 487 GB of disk is free.
- The 16-bit weights are already downloaded to `~/.cache/huggingface/hub/models--meta-llama--Llama-3.1-8B-Instruct` (15 GiB, 4 shards).
  4 stale `.incomplete` files remain in `blobs/` from the interrupted run.
- The KV cache for Llama 3.1 8B costs 128 KB per token at 16-bit.

## Decisions (from the grilling)

| # | Topic | Decision |
|---|---|---|
| 1 | Dependencies | `uv`, Python pinned to 3.12 |
| 2 | Layout | Small package `src/mlxbench/` plus thin CLI scripts in `scripts/` |
| 3 | Weights | Original 16-bit in the Hugging Face cache; 8-bit and 4-bit made with `mlx_lm convert` into `models/` (git-ignored) |
| 4 | Quality metric | Perplexity on WikiText-2 now; task benchmark (MMLU or similar) later |
| 5 | Speed metrics | Own harness on the `mlx-lm` Python API; separate timing for prefill (first token) and decode |
| 6 | Concurrency | Batched generation in one process: 1, 2, 4, 8 prompts |
| 7 | Lint | `ruff` plus `pre-commit` |
| 8 | Context lengths | 128, 1k, 4k, 16k tokens; concurrency only at 128 and 1k (8 x 16k cache would not fit) |
| 9 | Git | Private GitHub repo from the start |

## Steps (what each step does and why)

1. **Make the folder and run `git init`.**
   Why: all work needs one home under version control.
2. **Run `uv init` and pin Python 3.12 (`.python-version`).**
   Why: `uv` writes `pyproject.toml` and a lockfile.
   Same library versions on every run means benchmark numbers stay comparable.
3. **Add dependencies with `uv add`:** `mlx-lm`, `matplotlib`, `datasets` (for WikiText-2).
   Dev group: `ruff`, `pre-commit`, `pytest`.
   Why: `mlx-lm` runs and converts models; `datasets` fetches the test text; `matplotlib` draws charts.
4. **Write `.gitignore`:** `.venv/`, `models/`, `__pycache__/`, `.ruff_cache/`, raw output dumps.
   Why: keeps 29 GB of model files and secrets out of git.
   The Hugging Face token lives outside the repo, in `~/.cache/huggingface`.
5. **Create the package `src/mlxbench/`** with small modules (each under 500 lines, each function under 75 lines):
   - `loading.py`: load a model by path, return model and tokenizer.
   - `timing.py`: time prefill and decode separately, with warm-up and repeats.
   - `memory.py`: read peak memory with `mx.get_peak_memory()` and reset between runs.
   - `perplexity.py`: compute perplexity on fixed WikiText-2 chunks.
   - `batching.py`: batched generation for 1, 2, 4, 8 prompts.
   - `results.py`: save and load result records as JSON.
   Before writing `batching.py`, verify the batch API in the installed `mlx-lm` version.
6. **Create `scripts/`:** `quantize.sh` (calls `mlx_lm convert` for 8-bit and 4-bit), `bench.py` (runs the full grid, saves JSON to `results/`), `plot.py` (draws the trade-off charts).
   Why: scripts stay thin, so the logic is testable.
7. **Add `tests/`** with a few fast tests (timing and perplexity helpers on tiny inputs).
8. **Add `pyproject.toml` ruff config** (line limits, complexity and statement limits to flag long functions) and `.pre-commit-config.yaml` (ruff lint and format).
   Run `pre-commit install`.
   Never commit with `--no-verify`.
9. **Write `README.md`** and `docs/` (this plan, copied from the plan file with a topic name).
   Every file that holds code starts with a comment that has the last-edited date and time.
10. **First commit on a feature branch, then create a private GitHub repo with `gh repo create --private`.**
    Confirm the GitHub account or org with you first.
    Open a PR for the scaffold.
    Do not merge it.

## Defaults I will use unless you object

- Python 3.12, `matplotlib` for charts, results as JSON in `results/`.
- Model: `meta-llama/Llama-3.1-8B-Instruct` (Instruct variant).
- Quantization settings: group size 64 (MLX default), with `--q-bits 8` and `--q-bits 4`.
- Plots: speed vs memory vs perplexity, plus speed vs context length.

## Not in this scaffold

- Task benchmark (MMLU and similar): add after first results.
- Phase 2 (vLLM on a rented GPU).
- The Linear issue for this work: `linear-sync` will run at PR time.
  Outside `~/code/hemut3/`, I will ask which Linear workspace to use first.

## Verification

1. `uv sync` completes and `uv run python -c "import mlx.core as mx; print(mx.default_device())"` prints a GPU device.
2. `uv run pytest` passes.
3. `pre-commit run --all-files` passes.
4. `scripts/quantize.sh` makes `models/llama-3.1-8b-q8` and `models/llama-3.1-8b-q4`; their sizes are about 8.5 GB and 4.5 GB.
5. A one-model smoke run of `scripts/bench.py` (16-bit, 128 tokens, 1 prompt) prints tokens per second, time to first token, and peak memory, and writes a JSON file.
6. `scripts/plot.py` makes a chart from the smoke result.
7. After checks, stop any process I started so you have a clean machine.
