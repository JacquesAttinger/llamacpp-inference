# Last edited: 2026-10-14 11:35
"""Load a model, tokenize one prompt, and print the generated tokens one by one."""

from llamacppbench.loading import load_model

MODEL_PATH = "models/llama-3.1-8b-q8_0.gguf"
# MODEL_PATH = "models/llama-3.1-8b-f16.gguf"
PROMPT = (
    "Come up with one idea for a happy birthday text message for my mom's 54th birthday. "
    "Give only one, not several."
)
MAX_TOKENS = 100  # safety stop in case the model never emits an end-of-text token

model, tokenizer = load_model(MODEL_PATH)
prompt_tokens = tokenizer.encode(PROMPT)
print(f"Prompt: {PROMPT}")
print(f"Prompt tokens ({len(prompt_tokens)}): {prompt_tokens}")
print("Output: ", end="", flush=True)

# generate() is lazy: each pass through the loop asks the model for one more token.
for i, token in enumerate(model.generate(prompt_tokens, temp=0.0)):
    if token == model.token_eos() or i >= MAX_TOKENS:
        break
    print(model.detokenize([token]).decode(errors="replace"), end="", flush=True)
print()

model.close()
