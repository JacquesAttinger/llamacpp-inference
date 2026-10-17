<!-- Last edited: 2026-10-17 09:58 -->
<a id="readme-top"></a>

[![Forks][forks-shield]][forks-url]
[![Stargazers][stars-shield]][stars-url]
[![Issues][issues-shield]][issues-url]
[![MIT License][license-shield]][license-url]

<br />
<div align="center">
  <h3 align="center">llamacpp-bench</h3>

  <p align="center">
    Measure how fast Llama 3.1 8B runs on your Mac, and how much quantization changes that speed.
    <br />
    <br />
    <a href="https://github.com/JacquesAttinger/mlx-inference/issues">Report a problem or ask a question</a>
  </p>
</div>

<details>
  <summary>Table of Contents</summary>
  <ol>
    <li>
      <a href="#about-the-project">About The Project</a>
      <ul>
        <li><a href="#built-with">Built With</a></li>
      </ul>
    </li>
    <li>
      <a href="#getting-started">Getting Started</a>
      <ul>
        <li><a href="#prerequisites">Prerequisites</a></li>
        <li><a href="#installation">Installation</a></li>
      </ul>
    </li>
    <li><a href="#usage">Usage</a></li>
    <li><a href="#license">License</a></li>
    <li><a href="#contact">Contact</a></li>
    <li><a href="#acknowledgments">Acknowledgments</a></li>
  </ol>
</details>

## About The Project

![A run of scripts/run.py in the terminal][product-screenshot]

llamacpp-bench is a small project for learning how to benchmark LLM inference on Apple silicon.
It is built from scratch on [llama.cpp](https://github.com/ggml-org/llama.cpp), through the [llama-cpp-python](https://github.com/abetlen/llama-cpp-python) package.
The screenshot above is one run of `scripts/run.py`: the prompt, the model's reply, and the timings.

Quantization means storing each model weight with fewer bits, so the model file gets smaller.
The first experiment asks one question: how much faster does the model write text when the file is smaller?
Same prompt, same settings, on an Apple silicon Mac with 24 GB of memory:

| Model | File size | Mean decode speed | Speed-up over 16-bit |
| --- | --- | --- | --- |
| 16-bit | 16.1 GB | 8.6 tokens/s | 1.0x |
| 8-bit (`Q8_0`) | 8.5 GB | 16.0 tokens/s | 1.9x |
| 4-bit (`Q4_K_M`) | 4.9 GB | 26.9 tokens/s | 3.1x |

![Decode speed against bits per weight][experiment-plot]

A token is a small piece of text, often a short word or part of a word.
Decode speed is how many tokens the model writes per second after it has read the prompt.
The full method, the limits (one prompt, one machine) and the raw data are in [`results/Experiments/quantization_decode_tps/experiment.md`](results/Experiments/quantization_decode_tps/experiment.md).

Speed is only half of the story, so the second experiment checks quality.
It gives each model the test text of Wikitext-2 (a set of Wikipedia articles) and measures perplexity: how surprised the model is by the real next token.
Lower is better.

| Model | Perplexity | Change vs 16-bit |
| --- | --- | --- |
| 16-bit | 7.32 | 0 |
| 8-bit (`Q8_0`) | 7.33 | +0.05% |
| 4-bit (`Q4_K_M`) | 7.53 | +2.8% |

The 8-bit file is about twice as fast as 16-bit and scores almost the same.
The 4-bit file is three times as fast and scores 2.8% worse on this text.
Each model was run once, so a gap as small as 0.05% is below what this test can separate.
See [`results/Experiments/quantization_perplexity/experiment.md`](results/Experiments/quantization_perplexity/experiment.md) for the method and limits.

<p align="right">(<a href="#readme-top">back to top</a>)</p>

### Built With

* [![Python][Python-badge]][Python-url]
* [![uv][uv-badge]][uv-url]
* [![NumPy][NumPy-badge]][NumPy-url]
* [![Matplotlib][Matplotlib-badge]][Matplotlib-url]
* [![pytest][pytest-badge]][pytest-url]
* [![Ruff][Ruff-badge]][Ruff-url]
* [llama.cpp](https://github.com/ggml-org/llama.cpp) through [llama-cpp-python](https://github.com/abetlen/llama-cpp-python)

<p align="right">(<a href="#readme-top">back to top</a>)</p>

## Getting Started

### Prerequisites

* Python 3.12 or newer, and [uv](https://docs.astral.sh/uv/) to manage it.
* A Mac with Apple silicon. The numbers above come from a Mac with 24 GB of memory.
  The 16-bit model file is 16.1 GB, so a smaller Mac can run the 8-bit and 4-bit files only.
* The model files, in GGUF format (the file format llama.cpp reads).
  They are not in this repo, because they are too big and because Meta's Llama license applies to them.
  You make them yourself from Meta's weights in the next section.

### Installation

1. Clone the repo.
   ```sh
   git clone https://github.com/JacquesAttinger/mlx-inference.git
   cd mlx-inference
   ```
2. Install the Python packages.
   ```sh
   uv sync
   ```
3. Get access to [meta-llama/Llama-3.1-8B-Instruct](https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct) on Hugging Face and download it.
   This is a gated model, so you must accept Meta's terms first.
4. Build [llama.cpp](https://github.com/ggml-org/llama.cpp) in a folder of your choice.
   The script needs its `convert_hf_to_gguf.py` file and its `llama-quantize` program.
5. Make the three model files in `models/` (16-bit, `Q8_0` and `Q4_K_M`).
   The script that does this is kept in the `v0-bench-reference` tag:
   ```sh
   git show v0-bench-reference:scripts/quantize.sh > scripts/quantize.sh
   LLAMA_CPP_DIR=/path/to/llama.cpp MODEL=/path/to/Llama-3.1-8B-Instruct bash scripts/quantize.sh
   ```
   `MODEL` is the folder with `config.json` and the weight files.
   `LLAMA_CPP_DIR` must contain the built `llama-quantize` in `build/bin/` and a Python environment in `.venv/` that can run the convert script.

<p align="right">(<a href="#readme-top">back to top</a>)</p>

## Usage

Run this from the repo root, because the model and CSV paths are relative:

```sh
uv run python scripts/run.py
```

The script loads a model, feeds it one prompt, and prints the reply as the model writes it.
The reply sits under an `Output:` line and between quotation marks, so it is easy to tell from the other lines.
It stops at the end-of-text token or after 100 new tokens, whichever comes first.
Then it prints the timings and adds one row to `results/runs.csv`.

All flags are optional, and they can come in any order:

| Flag | Value | Default | What it does |
| --- | --- | --- | --- |
| `--tokens` | whole number, at least `1` | the prompt text once | Repeats the prompt text until it is exactly this many tokens long. Useful to see how prefill time grows with prompt size. |
| `--bits` | `4`, `8` or `16` | `8` | Picks the model file. |
| `--temp` | number, `0` or more | `0` | Sampling temperature. `0` always takes the most likely token, so the text is the same every time. A higher value, such as `0.8`, makes the text change from run to run. |
| `--seed` | whole number, `0` to `4294967294` | a random one | Repeats a run that uses a temperature above `0`. The same seed with the same model, prompt and settings gives the same text. It has no effect when `--temp` is `0`. |

```sh
uv run python scripts/run.py --tokens 1000 --bits 4 --temp 0.8 --seed 42
```

The `--bits` flag picks one of these files. The table is the `MODELS` dict at the top of `scripts/run.py`.
To add another model, add a line to that dict.

| `--bits` | File |
| --- | --- |
| `4` | `models/llama-3.1-8b-q4_k_m.gguf` |
| `8` | `models/llama-3.1-8b-q8_0.gguf` (the default) |
| `16` | `models/llama-3.1-8b-f16.gguf` |

The prompt text is the `PROMPT` constant in the same file.

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

### The results file

Each run adds a row to `results/runs.csv` with these columns:

- `model`: `4-bit`, `8-bit` or `16-bit`.
- `temp`, `seed`: the temperature and seed used. `seed` is empty in the oldest rows, which all used `temp` `0`.
- `top_k`, `top_p`, `min_p`, `typical_p`: the sampling settings used.
  The oldest rows show `40`, `0.95`, `0.05` and `1.0`, which were the library defaults that every run used before these settings existed.
- `prompt_tokens`, `generated_tokens`: how many tokens went in and came out.
- `prefill_s`: seconds the model spent reading the prompt, before it made the first token.
- `decode_s`: seconds it spent writing all the later tokens.
- `decode_tps`: tokens written per second during decode.

### Experiments

`results/Experiments/` has one folder per experiment.
Each folder has an `experiment.md` that describes it, the script that runs it, the CSV data, and a plot.
To repeat the first experiment, which takes a few minutes:

```sh
uv run python results/Experiments/quantization_decode_tps/run_experiment.py
```

The perplexity experiment downloads Wikitext-2 into `data/` on the first run (`data/` is not in git).
It takes about 26 minutes and uses most of the Mac's memory, because the 16-bit model is 16 GB:

```sh
uv run python results/Experiments/quantization_perplexity/run_experiment.py
```

### Project layout

- `src/llamacppbench/loading.py` has `load_model(path, n_ctx, logits_all, seed)`. It returns a `Llama` model and a `Tokenizer` with an `.encode(text)` method.
- `src/llamacppbench/sampling.py` reads and checks `sampling.toml` and `sampling.local.toml`.
- `src/llamacppbench/perplexity.py` has the chunking and the maths for the perplexity experiment.
- `scripts/run.py` is the command-line script described above.
- `tests/` has the tests. Run them with `uv run pytest`.

### Old version

An earlier, larger benchmark harness (timing, memory, perplexity, batching, plots) is kept for reference.
Read it with `git show v0-bench-reference:<path>` or `git checkout v0-bench-reference`.
The design notes are in [`docs/llama_cpp_conversion_plan.md`](docs/llama_cpp_conversion_plan.md).
The first design, which used Apple's MLX library, is in [`docs/mlx_inference_scaffold_plan.md`](docs/mlx_inference_scaffold_plan.md).

<p align="right">(<a href="#readme-top">back to top</a>)</p>

## License

Distributed under the MIT License. See [`LICENSE`](LICENSE) for more information.
The model files are not part of this repo, and Meta's Llama license applies to them.

<p align="right">(<a href="#readme-top">back to top</a>)</p>

## Contact

Jacques Attinger - [@JacquesAttinger](https://github.com/JacquesAttinger)

Project Link: [https://github.com/JacquesAttinger/mlx-inference](https://github.com/JacquesAttinger/mlx-inference)

<p align="right">(<a href="#readme-top">back to top</a>)</p>

## Acknowledgments

* [llama.cpp](https://github.com/ggml-org/llama.cpp) runs the models.
* [llama-cpp-python](https://github.com/abetlen/llama-cpp-python) gives Python access to llama.cpp.
* [Llama 3.1 8B Instruct](https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct) by Meta is the model that gets measured.
* [Best-README-Template](https://github.com/othneildrew/Best-README-Template) gave the structure of this README.

<p align="right">(<a href="#readme-top">back to top</a>)</p>

[forks-shield]: https://img.shields.io/github/forks/JacquesAttinger/mlx-inference.svg?style=for-the-badge
[forks-url]: https://github.com/JacquesAttinger/mlx-inference/network/members
[stars-shield]: https://img.shields.io/github/stars/JacquesAttinger/mlx-inference.svg?style=for-the-badge
[stars-url]: https://github.com/JacquesAttinger/mlx-inference/stargazers
[issues-shield]: https://img.shields.io/github/issues/JacquesAttinger/mlx-inference.svg?style=for-the-badge
[issues-url]: https://github.com/JacquesAttinger/mlx-inference/issues
[license-shield]: https://img.shields.io/github/license/JacquesAttinger/mlx-inference.svg?style=for-the-badge
[license-url]: https://github.com/JacquesAttinger/mlx-inference/blob/main/LICENSE
[product-screenshot]: images/screenshot.png
[experiment-plot]: results/Experiments/quantization_decode_tps/decode_tps_vs_bits.png
[Python-badge]: https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white
[Python-url]: https://www.python.org/
[uv-badge]: https://img.shields.io/badge/uv-DE5FE9?style=for-the-badge&logo=uv&logoColor=white
[uv-url]: https://docs.astral.sh/uv/
[NumPy-badge]: https://img.shields.io/badge/NumPy-013243?style=for-the-badge&logo=numpy&logoColor=white
[NumPy-url]: https://numpy.org/
[Matplotlib-badge]: https://img.shields.io/badge/Matplotlib-11557c?style=for-the-badge&logo=plotly&logoColor=white
[Matplotlib-url]: https://matplotlib.org/
[pytest-badge]: https://img.shields.io/badge/pytest-0A9EDC?style=for-the-badge&logo=pytest&logoColor=white
[pytest-url]: https://docs.pytest.org/
[Ruff-badge]: https://img.shields.io/badge/Ruff-D7FF64?style=for-the-badge&logo=ruff&logoColor=black
[Ruff-url]: https://docs.astral.sh/ruff/
