---
name: new-experiment
description: Set up, run and write up a reproducible benchmark experiment in results/Experiments/. Use when the user describes an experiment to run on the local llama.cpp models (for example "measure how quantization changes decode speed", "compare prefill time across prompt lengths"). First asks the user about anything the description leaves open, then builds the experiment folder (md description, run script, CSV data, optional plot), runs it, and reports the results.
---
<!-- Last edited: 2026-10-18 09:00 -->

# New experiment

An experiment lives in its own folder, `results/Experiments/<snake_case_name>/`.
Anyone must be able to rerun it from that folder's script and get the same kind of numbers.
The first example is `results/Experiments/quantization_decode_tps/`. Read its files before you start, and copy its shape.

## 1. Get current, then read what exists

1. Run `git fetch` and fast-forward `main`. If the working tree is dirty and the fast-forward would conflict, stop and tell the user.
   Uncommitted changes that do not conflict (for example new rows in `results/runs.csv`) are fine. Leave them alone.
2. Read `scripts/run.py`. Reuse what is there instead of writing new code:
   - `MODELS` maps bits per weight (4, 8, 16) to a model file.
   - `PROMPT` is the default prompt. `make_prompt()` builds a prompt of an exact token count.
   - `generate_timed(model, prompt_tokens)` times prefill and decode and returns `generated_tokens`, `prefill_s`, `decode_s` and `decode_tps`.
   - `MAX_TOKENS` and `MIN_N_CTX` are the token cap and the base context size.
3. Read `results/Experiments/quantization_decode_tps/run_experiment.py` and `experiment.md`.
4. Look at `results/Experiments/` to see what exists, so the new name does not clash.

## 2. Ask about every open question before you write code

The user gives a short description. Work out what it does not say.
Ask only about what is open. Do not ask about anything the user already said.
Use `AskUserQuestion`, at most 4 questions per call, with your recommended default as the first option.
If the answers raise new questions, ask again. Do not guess at something that changes the numbers.

Check each item on this list:

- **Question and metric.** What is the experiment trying to learn?
  Which number answers it (`decode_tps`, `prefill_s`, something new)?
  Prefer a rate such as `decode_tps` over a total such as `decode_s`, because totals change with how many tokens the model wrote.
  Say so if the user picks a total.
- **What changes and what stays fixed.** The variable (bits, prompt length, token cap, ...) and its values.
  Everything else must stay the same across the groups.
- **Models.** Which entries of `MODELS`?
  Default: the 8-bit model (`DEFAULT_BITS` in `run.py`), unless the experiment is about the model itself (for example quantization).
  If the user names a model file that has no entry in `MODELS`, ask for its bit label and add it there.
- **Prompt.** The default `PROMPT`, a different text, or an exact token count (`make_prompt`)?
- **Trials.** How many trials per group, and how many are warm-up that get dropped?
  Mean or median of the rest? Load each model once, or reload before every trial?
  Default: load once, drop the first 2 of 7, take the mean of the other 5.
- **Order.** Which group runs first? Order can matter, because of heat and memory.
- **Generation settings.** Token cap and sampling. Default: `MAX_TOKENS` and greedy (`temp=0`), as in `generate_timed`.
- **Outputs.** CSV columns, and whether the user wants a plot.
  Make a plot only if the user asks for one. If they do, ask what goes on each axis.
- **New code.** Does the metric need something `run.py` cannot give (for example peak memory or perplexity)?
  If so, say what you plan to add and where, and get a yes before you start.

Then write a short plan in your reply: folder name, the groups, the trial scheme, the metric, the outputs.
Wait for a yes if anything in it was not stated by the user.

## 3. Build the folder

Create `results/Experiments/<name>/` with these files.
Put a "last edited" date and time comment at the top of every file you write.

- `experiment.md`: the description. Start with a TLDR a 15 year old understands. Then Question, Method, How to run it again, Results, What we see, Limits.
  Put each sentence on its own line. Use short, plain sentences (ASD-STE100).
  Write Results and What we see only after the run.
- `run_experiment.py`: the script that reproduces everything. It must:
  - Say in its docstring what it does and how to run it.
  - Find the repo root from `__file__`, add `scripts/` to `sys.path`, then `os.chdir` to the repo root. The model paths in `run.py` are relative.
  - Import from `run.py` (`MODELS`, `PROMPT`, `generate_timed`, ...). Do not copy the timing code.
  - Run each group, print one line per trial as it goes, then free the model (`model.close()`, `del`, `gc.collect()`) before the next one loads.
  - Write `trials.csv` with one row per trial and a `phase` column (`warmup` or `measured`). Keep warm-up rows in the file.
  - Write `summary.csv` with the mean and the sample standard deviation per group.
  - Have a `--plot-only` flag that redraws the plot from the saved CSVs and runs no model, if there is a plot.
- `trials.csv` and `summary.csv`: written by the script.
- `<metric>_vs_<variable>.png`: only if the user asked. Use matplotlib with the `Agg` backend.
  Show the group mean with the standard deviation as error bars, and the individual trials as grey dots.
  Put values that grow by powers of 2 (bits, context size) on a log2 axis so they are evenly spaced.
  Label the axes with units, and label each mean on the plot.
  Look at the finished image yourself and fix anything that looks off.

Keep the repo rules: no file over 500 lines, no function over 75 lines, line length 110.
If an experiment needs a change to `scripts/run.py`, make it a small refactor that keeps `run.py` working the same, and say so in the PR.

## 4. Check, then run

1. Run `uv run ruff check` and `uv run ruff format` on the new folder, and fix every finding.
2. Run the script's `--help` to confirm it imports.
3. Before a run that loads a big model, tell the user that the run uses all the Mac's memory and takes minutes. The 16-bit model is 16 GB on a 24 GB machine.
4. Run the experiment in the background (`run_in_background`) and wait for the notification. Do not poll.
5. Read `trials.csv` and `summary.csv`. Check that every trial wrote the expected number of tokens and that nothing looks broken (a group that is far off, a missing row).
6. Fill in Results, What we see and Limits in `experiment.md`.
   Only claim what the numbers show. Mark an explanation you did not measure as "our reading", for example "memory speed is the limit".
   List the limits honestly: one prompt, one machine, one run of the whole experiment, anything not measured.
7. Make sure no process or port from the run is still open.

## 5. Report

Reply with a TLDR first, then the results table, the files you made, and the caveats.
Do not commit or open a PR until the user asks.
When they do:

- Branch from the fresh `main` (`feat/<name>`). Add only the experiment folder, `README.md` and any `run.py` change.
- Leave out unrelated changes such as `results/runs.csv` rows and `.DS_Store` files. Say that you left them out.
- Add a line about the experiment to the "What is here" section of `README.md` if it is not already there.
- Write a short PR description with the results table. Do not add yourself as co-author.
- Merge only if the user said to merge. Delete the branch after.
