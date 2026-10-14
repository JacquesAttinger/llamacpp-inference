# Last edited: 2026-10-14 09:40
"""Batched generation with llama.cpp's low-level batch API (llama_cpp.llama_decode)."""

import time

import llama_cpp

from llamacppbench.memory import peak_gb, reset_peak
from llamacppbench.timing import TimingResult


def _add_token(batch, i: int, token: int, pos: int, seq: int, want_logits: bool) -> None:
    batch.token[i] = token
    batch.pos[i] = pos
    batch.n_seq_id[i] = 1
    batch.seq_id[i][0] = seq
    batch.logits[i] = want_logits


def _make_context(model, batch_size: int, n_prompt: int, max_tokens: int):
    params = llama_cpp.llama_context_default_params()
    params.n_seq_max = batch_size
    params.n_ctx = batch_size * (n_prompt + max_tokens)
    params.n_batch = 2048
    params.n_ubatch = 512
    return llama_cpp.llama_init_from_model(model._model.model, params)


def _prefill(ctx, batch, smpl, prompt_tokens: list[int], batch_size: int) -> list[int]:
    n_prompt = len(prompt_tokens)
    first_tokens = []
    for seq in range(batch_size):
        for i, token in enumerate(prompt_tokens):
            _add_token(batch, i, token, i, seq, want_logits=(i == n_prompt - 1))
        batch.n_tokens = n_prompt
        llama_cpp.llama_decode(ctx, batch)
        first_tokens.append(llama_cpp.llama_sampler_sample(smpl, ctx, n_prompt - 1))
    return first_tokens


def _decode_step(ctx, batch, smpl, tokens: list[int], n_prompt: int, step: int) -> list[int]:
    for seq, token in enumerate(tokens):
        _add_token(batch, seq, token, n_prompt + step, seq, want_logits=True)
    batch.n_tokens = len(tokens)
    llama_cpp.llama_decode(ctx, batch)
    return [llama_cpp.llama_sampler_sample(smpl, ctx, seq) for seq in range(len(tokens))]


def time_batch(
    model, tokenizer, prompt_tokens: list[int], batch_size: int, max_tokens: int = 128
) -> TimingResult:
    """Run batch_size copies of one prompt; report aggregate tokens per second."""
    n_prompt = len(prompt_tokens)
    reset_peak()
    ctx = _make_context(model, batch_size, n_prompt, max_tokens)
    batch = llama_cpp.llama_batch_init(max(n_prompt, batch_size), 0, batch_size)
    smpl = llama_cpp.llama_sampler_chain_init(llama_cpp.llama_sampler_chain_default_params())
    llama_cpp.llama_sampler_chain_add(smpl, llama_cpp.llama_sampler_init_greedy())
    try:
        t0 = time.perf_counter()
        tokens = _prefill(ctx, batch, smpl, prompt_tokens, batch_size)
        t1 = time.perf_counter()
        for step in range(max_tokens - 1):
            tokens = _decode_step(ctx, batch, smpl, tokens, n_prompt, step)
        t2 = time.perf_counter()
    finally:
        llama_cpp.llama_sampler_free(smpl)
        llama_cpp.llama_batch_free(batch)
        llama_cpp.llama_free(ctx)
    prefill_tps = batch_size * n_prompt / (t1 - t0)
    decode_tps = batch_size * (max_tokens - 1) / (t2 - t1)
    return TimingResult(prefill_tps, decode_tps, t1 - t0, peak_gb())
