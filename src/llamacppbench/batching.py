# Last edited: 2026-10-13 11:10
"""Batched generation with llama.cpp's low-level batch API (llama_cpp.llama_decode)."""

from llamacppbench.timing import TimingResult


def time_batch(
    model, tokenizer, prompt_tokens: list[int], batch_size: int, max_tokens: int = 128
) -> TimingResult:
    """Run batch_size copies of one prompt; report aggregate tokens per second."""
    # TODO(human): write the batched decode loop. There is no one-call batch_generate in llama-cpp-python,
    # so you drive llama.cpp yourself. Steps, in order (the names are all in the `llama_cpp` module):
    #   1. Make a new context that can hold batch_size sequences. `Llama` cannot set this, so:
    #        p = llama_context_default_params(); p.n_seq_max = batch_size
    #        p.n_ctx = batch_size * (len(prompt_tokens) + max_tokens)   # n_ctx is shared by all sequences
    #        p.n_batch = 2048; p.n_ubatch = 512
    #        ctx = llama_init_from_model(model._model.model, p)          # reuses the already-loaded weights
    #   2. batch = llama_batch_init(n_tokens, 0, batch_size) and a greedy sampler:
    #        smpl = llama_sampler_chain_init(llama_sampler_chain_default_params())
    #        llama_sampler_chain_add(smpl, llama_sampler_init_greedy())
    #   3. Prefill: for each sequence s, fill the batch with the prompt (token, pos, n_seq_id=1, seq_id[0]=s,
    #      logits=True only on the last token), call llama_decode(ctx, batch), then take the first new token
    #      with llama_sampler_sample(smpl, ctx, len(prompt_tokens) - 1)  (right after that sequence's decode).
    #   4. Decode loop: each step, one batch with one token per sequence (pos = how many tokens it has so far,
    #      logits=True on all), llama_decode, then llama_sampler_sample(smpl, ctx, i) for sequence i.
    #   5. Return TimingResult(prefill_tps, decode_tps, ttft, peak_gb()). Count tokens across ALL sequences.
    # Gotchas I checked on this machine:
    #   - Time prefill and decode yourself with time.perf_counter(). llama_perf_context counts a decode
    #     step of batch_size tokens as "prompt" tokens, so its numbers are wrong for batches.
    #   - Free everything in a `finally`: llama_sampler_free, llama_batch_free, llama_free. If you do not,
    #     the process aborts at exit (a ggml Metal assert).
    #   - Call reset_peak() from llamacppbench.memory first, like the old code did.
    raise NotImplementedError("time_batch is a TODO(human); run bench.py with --batches 1 until it is done")
