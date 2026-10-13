<!-- Last edited: 2026-10-13 11:25 -->
# Plan: convert the benchmark from MLX to llama.cpp

## TLDR

Right now the project runs Llama 3.1 8B through MLX, Apple's own library.
We will swap the engine under it for llama.cpp, and keep using Python through the `llama-cpp-python` package.
About half of your code (the results file, the plots, the median-of-runs logic, the perplexity math, the test grid) stays as it is.
The other half (loading, memory, batching, quantizing) gets new insides, but keeps the same function names, so the rest of the code does not notice.

## What stays, what changes

| File | Verdict | Why |
| --- | --- | --- |
| `src/mlxbench/results.py` | Keep | It only stores numbers in JSON. It knows nothing about MLX. |
| `scripts/plot.py` | Keep | It only reads `BenchRecord`s. |
| `tests/test_helpers.py` | Keep | All four tests test pure math and JSON, not MLX. They must still pass after the swap. |
| `src/mlxbench/timing.py` | Keep most | `TimingResult`, `summarize`, and `time_generation` (warm up, then take the median) are engine-free. Only the body of `time_once` changes. |
| `src/mlxbench/perplexity.py` | Keep most | `perplexity_from_nll`, `chunk_tokens`, and `wikitext2_perplexity` are engine-free. `chunk_nll` is rewritten with numpy. `wikitext2_tokens` gets a new tokenizer call. |
| `scripts/bench.py` | Keep most | The grid, `make_prompt`, and the record building stay. Only the load call and the default model paths change. |
| `src/mlxbench/loading.py` | Rewrite | `mlx_lm.load` becomes `llama_cpp.Llama(...)`. |
| `src/mlxbench/memory.py` | Rewrite | `mx.get_peak_memory` has no llama.cpp twin. See step 5. |
| `src/mlxbench/batching.py` | Rewrite | `batch_generate` has no high-level twin. This is the biggest piece of new learning. |
| `scripts/quantize.sh` | Rewrite | `mlx_lm.convert` becomes llama.cpp's `convert_hf_to_gguf.py` plus `llama-quantize`. |
| `pyproject.toml`, `README.md`, plan doc | Edit | Swap `mlx-lm` for `llama-cpp-python`, and update the words. |

## Steps

### 1. Install the tools (about 15 minutes)

- `cmake` is not installed on this machine yet.
  Run `brew install cmake`.
- Clone `llama.cpp` next to this repo (not inside it).
  Build it with `cmake -B build && cmake --build build -j`.
  On Apple silicon the Metal GPU backend is on by default.
  This gives us `llama-quantize`, `llama-perplexity`, and `llama-batched-bench`.
  We use them to check our Python numbers.
- Add `llama-cpp-python` to `pyproject.toml` and run `uv sync`.
  It compiles llama.cpp inside the install, so it needs `cmake` and the Xcode command line tools.
  Check the build log for Metal.
- Do not remove `mlx-lm` yet.
  It is useful for a side-by-side check until step 9.

### 2. Make the GGUF files (replaces `scripts/quantize.sh`)

- llama.cpp does not read MLX folders.
  It reads one `.gguf` file per model.
- Convert the local Hugging Face snapshot (the same path as `DEFAULT_MODEL` today) with `convert_hf_to_gguf.py --outtype f16`.
  This is the 16-bit baseline. It is about 16 GB on disk.
- Run `llama-quantize` on it to make `Q8_0` and `Q4_K_M`.
  Output goes to `models/`, which is already in `.gitignore`.
- Keep the same "skip if the output exists" check the script has now.
- Caveat: these quant types are not the same as your MLX ones.
  MLX 4-bit is one plain scheme with group size 64.
  `Q4_K_M` mixes 4-bit and 6-bit blocks, so it is slightly bigger and slightly more accurate.
  We could also make `Q4_0`, which is closer to MLX 4-bit.
  The numbers from the two engines are a comparison of two designs, not a controlled test.

### 3. Smoke test: one prompt, one answer

- Write a throwaway script (not part of the package) that loads the f16 GGUF with `Llama(model_path=..., n_gpu_layers=-1, n_ctx=2048)`.
  `n_gpu_layers=-1` means "put every layer on the GPU".
- Generate 20 tokens and print them.
- Confirm in the load log that Metal is on and all layers are offloaded.
  If they are not, every later speed number is wrong, so we stop here until it works.

### 4. Port `loading.py`

- `load_model` builds a `Llama` object.
  It takes `n_ctx`, `n_seq_max`, and `logits_all` as arguments, because perplexity and batching need different settings (see steps 6 and 7).
- In MLX the tokenizer was a separate object.
  In llama.cpp the model owns the tokenizer.
- To keep `bench.py` and `perplexity.py` working with little change, add a tiny `Tokenizer` wrapper with an `.encode(text) -> list[int]` method.
  It calls `llm.tokenize(text.encode(), add_bos=...)`.
  Then `make_prompt` in `bench.py` does not change at all.
- Update `DEFAULT_MODEL` to point to the f16 GGUF file.

### 5. Port `memory.py` (a design choice for you)

- MLX counts its own GPU memory and lets you reset the counter.
  llama.cpp has no such counter.
- Option A (my pick): run each measured run in a fresh subprocess, and read the child's peak resident memory with `resource.getrusage(RUSAGE_CHILDREN).ru_maxrss`.
  On macOS this value is in bytes.
  It is cheap and honest, and a fresh process also means a fresh peak per run.
- Option B: compute it from sizes (model file size plus KV cache size).
  It is predictable, but it is an estimate and not a measurement.
- Caveat for either option: the Mac shares one pool of memory between CPU and GPU, and llama.cpp memory-maps the weights.
  So the number is "memory the process holds", which is close to, but not exactly, what MLX reported.
- Keep the names `reset_peak()` and `peak_gb()` if Option A allows it.
  If it does not, change `timing.py` to match.

### 6. Port `timing.py:time_once`

- Keep `TimingResult`, `summarize`, and `time_generation` exactly.
- llama.cpp keeps its own timers, like `mlx-lm` did.
  The low-level binding exposes `llama_perf_context` with fields such as `t_p_eval_ms` and `n_p_eval` (prompt) and `t_eval_ms` and `n_eval` (generation).
  Reset them before each run, read them after.
- Then `prefill_tps = n_p_eval / (t_p_eval_ms / 1000)`, `decode_tps` likewise, and `ttft = prompt_tokens / prefill_tps`.
  That is the same formula you already have, so your comments stay true.
- Use greedy sampling (`temperature=0`) so every run does the same work.
- I have not yet checked the exact field names in 0.3.36.
  Step 3 will confirm them before we write this.

### 7. Port `perplexity.py:chunk_nll`

- Load a second `Llama` with `logits_all=True`, so it keeps the scores for every position, not just the last.
- For each chunk: reset the KV cache, call `eval(chunk)`, and read the scores.
  Then a log-softmax in numpy gives the negative log-likelihood of each next token.
  This replaces `nn.losses.cross_entropy`.
- `perplexity_from_nll`, `chunk_tokens`, and `wikitext2_perplexity` do not change.
- Memory check: 1024 positions times 128,256 vocab words times 4 bytes is about 0.5 GB per chunk.
  That is fine.
- Cross-check against the `llama-perplexity` binary.
  The two will not match exactly, because it scores each chunk differently, but they should be close.
- Caveat: llama.cpp tokenizes the text in its own way.
  The token counts may differ slightly from your MLX run, so the perplexity values may differ a little for that reason alone.

### 8. Port `batching.py` (the big learning step)

- `llama-cpp-python` has no one-call `batch_generate`.
- Use the low-level API: create the model with `n_seq_max=B`, then build an `llama_batch` that holds B sequences with different sequence ids and call `llama_decode`.
  Do the prefill for all B, then loop: take the highest-scoring token for each sequence, feed them back as the next batch, and stop after `max_tokens`.
- Note that `n_ctx` is shared by all sequences in llama.cpp.
  For B sequences of 1024 tokens plus 128 generated, you need `n_ctx` of about `B * 1152`.
  The existing `BATCH_MAX_CONTEXT = 1024` rule still fits.
- Time it with the same perf counters, and return the same `TimingResult` so `bench.py` does not change.
- Cross-check with `llama-batched-bench`.
- Fallback if this gets too deep: run `llama-server` with parallel slots and hit it over HTTP.
  That changes what we measure, so I would only do it if you want to.

### 9. Wire up `bench.py` and clean up

- Change the `--models` defaults to the three GGUF files.
- Add `--backend` or, simpler, add a `quant` or `backend` field to `BenchRecord` (it has a default, so old JSON files still load).
  That lets one `plot.py` chart show old MLX results and new llama.cpp results together.
- Run the small command from the README first (`--no-ppl --contexts 128 --batches 1`), then the full grid.
- Run `pytest` and `ruff`.
  All four existing tests should pass without edits.
- Update `README.md` and the scaffold plan doc.
- Decide on the package name (see questions).
- Once you are happy, remove `mlx-lm` from `pyproject.toml`.

## Process

- Work on a feature branch and open a PR.
  We do not commit to `main`.
- One commit per step above, so each step can be reviewed and undone alone.
- Pre-commit hooks stay on.

## Where I would ask you to write the code

These are the parts with real design choices and real learning:
- Step 5: how to measure peak memory (your pick between A and B).
- Step 7: `chunk_nll` with numpy.
- Step 8: the batched decode loop.

I would write the plumbing (steps 1 to 4, 6, and 9) and leave those three for you, with a stub and a `TODO(human)` marker for each.

## Decisions (answered)

1. Replace MLX completely. The old engine is gone from the code. No `backend` field is needed in `BenchRecord`.
2. Rename the package to `llamacppbench`. The Git repo folder and the GitHub repo name are still `mlx-inference`. Rename those by hand when you are ready.
3. Quant types: `Q8_0` and `Q4_K_M`.

## What the first build found

- `llama.cpp` built with Metal. The model loads with 33 of 33 layers on the GPU. The GPU memory limit on this Mac is about 18 GB.
- `llama_perf_context` has the fields `t_p_eval_ms`, `t_eval_ms`, `n_p_eval`, and `n_eval`. The speeds from `time_once` match `llama-bench` to within about 3%.
- `llama-cpp-python` reuses the KV cache when the same prompt comes twice, so `time_once` calls `model.reset()` first. Without that the prefill is skipped and the speed number is fake.
- `Llama(...)` cannot set `n_seq_max`. The batching code must build its own context.
- The perf counters count every multi-token decode step as prompt tokens. So batched runs need their own `time.perf_counter()` timing.
- The old perplexity code asked for the dataset `wikitext`. `datasets` 5.x needs `Salesforce/wikitext`. This is fixed.
- The f16 model is 15 GB. With the 18 GB GPU limit, the 16k-context f16 run is tight. It may fail or slow down.

## Risks

- `llama-cpp-python` follows llama.cpp with a delay, and its low-level function names can change between versions.
  Pin the version in `pyproject.toml`.
- A source build can fail on a new macOS or Xcode.
  Step 1 is where we find out.
- The memory numbers (step 5) are the least exact of all the measurements.
  We should say so in the README.
