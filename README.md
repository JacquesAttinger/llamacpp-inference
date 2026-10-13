<!-- Last edited: 2026-10-13 11:25 -->
# llamacpp-bench

Benchmarks Llama 3.1 8B Instruct on Apple silicon with llama.cpp (through `llama-cpp-python`).
It compares 16-bit, Q8_0, and Q4_K_M weights on speed, memory, and perplexity.

## Setup

```bash
brew install cmake
git clone https://github.com/ggml-org/llama.cpp ~/code/llama.cpp
cmake -S ~/code/llama.cpp -B ~/code/llama.cpp/build && cmake --build ~/code/llama.cpp/build -j
uv venv ~/code/llama.cpp/.venv && uv pip install --python ~/code/llama.cpp/.venv/bin/python \
  -r ~/code/llama.cpp/requirements/requirements-convert_hf_to_gguf.txt   # only the converter needs this

uv sync
uv run pre-commit install
scripts/quantize.sh          # makes models/llama-3.1-8b-{f16,q8_0,q4_k_m}.gguf
```

Set `LLAMA_CPP_DIR` if llama.cpp is not in `~/code/llama.cpp`.

## Run

```bash
uv run python scripts/bench.py --no-ppl --models models/llama-3.1-8b-q4_k_m.gguf --contexts 128 --batches 1
uv run python scripts/bench.py   # full grid
uv run python scripts/plot.py    # results/bench.json -> results/bench.png
```

To cross-check the speeds, run `~/code/llama.cpp/build/bin/llama-bench -m models/llama-3.1-8b-q4_k_m.gguf -p 128,1024 -n 128 -ngl 99 -fa 1`.

See `docs/llama_cpp_conversion_plan.md` for the design.
The first design (MLX) is in `docs/mlx_inference_scaffold_plan.md`.
