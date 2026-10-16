<!-- Last edited: 2026-10-16 07:45 -->
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
The reply sits under an `Output:` line and between quotation marks, so it is easy to tell from the other lines.
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

To set the sampling temperature, pass `--temp`.
The default is `0`, which always takes the most likely token, so the text is the same every time.
A higher value, such as `0.8`, picks the next token at random from the likely ones, so the text changes from run to run.
A negative value is an error.

```bash
uv run python scripts/run.py --temp 0.8
```

To repeat a run that uses a temperature above `0`, pass `--seed`.
The seed starts the random number generator that makes the picks.
The same seed with the same model, prompt and settings gives the same text.
It must be a whole number from `0` to `4294967294`.
If you leave it out, the script picks a random seed and writes it to the CSV, so you can repeat that run later.
The seed has no effect when `--temp` is `0`.

```bash
uv run python scripts/run.py --temp 0.8 --seed 42
```

### Flags together

All flags are optional, and they can come in any order.
This README uses one order, the same as `--help` shows them: `--tokens`, `--bits`, `--temp`, `--seed`.

```bash
uv run python scripts/run.py --tokens 1000 --bits 4 --temp 0.8 --seed 42
```

| Flag | Value | Default |
| --- | --- | --- |
| `--tokens` | whole number, at least `1` | the prompt text once |
| `--bits` | `4`, `8` or `16` | `8` |
| `--temp` | number, `0` or more | `0` |
| `--seed` | whole number, `0` to `4294967294` | a random one |

### Sampling settings

Four more settings decide which words the model may pick: `top_k`, `top_p`, `min_p` and `typical_p`.
They are not flags. They live in `sampling.toml` in the repo root, with a comment on each one that says what it does, its range and some sensible values.
The model applies these filters first, and `--temp` reshuffles the words that are left.
So with the default filters, even a very high `--temp` still gives readable text.
To see the full effect of `--temp`, turn the filters off.

To try other values without changing git, make a file called `sampling.local.toml` next to `sampling.toml`.
Git ignores it. List only the settings you want to change:

```toml
top_k = 0
top_p = 1.0
min_p = 0.0
```

When that file changes a value, `run.py` prints the changes on the first line of the run.
A wrong value, such as `top_p = 3`, stops the run with an error that names the file and the setting.
Delete the file to go back to the committed values.

Each run adds a row with these columns:

- `temp`: the temperature used.
- `seed`: the seed used. It is empty in the oldest rows, which were all run with `temp` `0`.
- `top_k`, `top_p`, `min_p`, `typical_p`: the sampling settings used.
  The oldest rows show `40`, `0.95`, `0.05` and `1.0`, which were the library defaults that every run used before these settings existed.
- `prefill_s`: seconds the model spent reading the prompt, before it made the first token.
- `decode_s`: seconds it spent writing all the later tokens.
- `decode_tps`: tokens written per second during decode.

## What is here

`src/llamacppbench/loading.py` has `load_model(path, n_ctx, logits_all, seed)`.
It returns a `Llama` model and a `Tokenizer` with an `.encode(text)` method.

`src/llamacppbench/sampling.py` reads and checks `sampling.toml` and `sampling.local.toml`.

`scripts/run.py` is the command-line script described above.

`results/Experiments/` has one folder per experiment.
Each folder has an `experiment.md` that describes it, the script that runs it, the CSV data, and a plot.
The first one is `quantization_decode_tps/`.

## Old version

The first full benchmark harness (timing, memory, perplexity, batching, plots) is kept for reference.
Read it with `git show v0-bench-reference:<path>` or `git checkout v0-bench-reference`.

See `docs/llama_cpp_conversion_plan.md` for the old design.
The first design (MLX) is in `docs/mlx_inference_scaffold_plan.md`.
