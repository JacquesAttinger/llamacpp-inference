# Last edited: 2026-10-18 09:20
"""Generate for several prompts at once with llama.cpp's own batching (one sequence id per prompt).

llama-cpp-python's Llama.generate handles only one prompt, so this talks to its low-level API.
Every step feeds one token per prompt into a single llama_decode call, so the model weights are read once
per step for the whole batch. Sampling is greedy, and end-of-text is ignored so every prompt writes
exactly `n_new` tokens and the batch never shrinks.
"""

import time

import llama_cpp
from llama_cpp import _internals as internals

N_CTX = 2048  # one KV cache shared by all sequences (kv_unified), so 7 x ~125 tokens fit easily
N_BATCH = 2048  # most tokens one llama_decode call can take (the prefill of 7 prompts is ~175)
N_UBATCH = 512  # same as llamacppbench.loading.load_model


class BatchedModel:
    """A model loaded for up to n_seq_max prompts in flight at once."""

    def __init__(self, path: str, n_seq_max: int):
        model_params = llama_cpp.llama_model_default_params()
        model_params.n_gpu_layers = 0x7FFFFFFF  # every layer on the GPU (Metal), as in load_model
        model_params.use_mmap = False
        self.model = internals.LlamaModel(path_model=path, params=model_params, verbose=False)
        params = llama_cpp.llama_context_default_params()
        params.n_ctx = N_CTX
        params.n_batch = N_BATCH
        params.n_ubatch = N_UBATCH
        params.n_seq_max = n_seq_max
        params.kv_unified = True
        params.flash_attn_type = llama_cpp.LLAMA_FLASH_ATTN_TYPE_ENABLED
        self.ctx = internals.LlamaContext(model=self.model, params=params, verbose=False)
        self.batch = internals.LlamaBatch(n_tokens=N_BATCH, embd=0, n_seq_max=1, verbose=False)
        self.sampler = llama_cpp.llama_sampler_chain_init(llama_cpp.llama_sampler_chain_default_params())
        llama_cpp.llama_sampler_chain_add(self.sampler, llama_cpp.llama_sampler_init_greedy())

    def tokenize(self, text: str) -> list[int]:
        return list(self.model.tokenize(text.encode(), add_bos=True, special=False))

    def _pick(self, index: int) -> int:
        """Greedy token from the index-th output slot of the last llama_decode call."""
        return llama_cpp.llama_sampler_sample(self.sampler, self.ctx.ctx, index)

    def generate(self, prompt_tokens: list[int], n_prompts: int, n_new: int) -> dict:
        """Run n_prompts copies of one prompt together, writing n_new tokens each. Times both phases."""
        self.ctx.kv_cache_clear()  # forget the cached prompts, so every call pays for prefill again
        batch = self.batch
        batch.reset()
        for seq_id in range(n_prompts):
            batch.add_sequence(prompt_tokens, seq_id, logits_all=False)  # only the last token gets scores
        # Prefill: all prompts go through in one llama_decode call, then token 1 of each is picked.
        start = time.perf_counter()
        self.ctx.decode(batch)
        last = [self._pick(i * len(prompt_tokens) + len(prompt_tokens) - 1) for i in range(n_prompts)]
        prefill_s = time.perf_counter() - start

        # Decode: each step feeds the newest token of every prompt, one llama_decode call for all of them.
        start = time.perf_counter()
        for step in range(n_new - 1):
            batch.reset()
            batch.batch.n_tokens = n_prompts
            for seq_id in range(n_prompts):
                batch.batch.token[seq_id] = last[seq_id]
                batch.batch.pos[seq_id] = len(prompt_tokens) + step
                batch.batch.seq_id[seq_id][0] = seq_id
                batch.batch.n_seq_id[seq_id] = 1
                batch.batch.logits[seq_id] = True
            self.ctx.decode(batch)
            last = [self._pick(i) for i in range(n_prompts)]
        decode_s = time.perf_counter() - start
        return {"prefill_s": prefill_s, "decode_s": decode_s, "last_tokens": last}

    def close(self) -> None:
        llama_cpp.llama_sampler_free(self.sampler)
        self.batch.close()
        self.ctx.close()
        self.model.close()
