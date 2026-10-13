#!/usr/bin/env bash
# Last edited: 2026-10-13 10:30
# Make GGUF copies of the model: 16-bit baseline, then Q8_0 and Q4_K_M (llama.cpp quantization).
# Needs a built llama.cpp checkout (see README). Override its location with LLAMA_CPP_DIR.
set -euo pipefail
cd "$(dirname "$0")/.."
LLAMA_CPP_DIR="${LLAMA_CPP_DIR:-$HOME/code/llama.cpp}"
# A folder with config.json and the safetensors shards (the local Hugging Face snapshot).
MODEL="${MODEL:-$HOME/.cache/huggingface/hub/models--meta-llama--Llama-3.1-8B-Instruct/snapshots/0e9e39f249a16976918f6564b8830bc894c89659}"
mkdir -p models
f16="models/llama-3.1-8b-f16.gguf"

if [ -f "$f16" ]; then
  echo "skip $f16 (exists)"
else
  "$LLAMA_CPP_DIR/.venv/bin/python" "$LLAMA_CPP_DIR/convert_hf_to_gguf.py" "$MODEL" --outtype f16 --outfile "$f16"
fi

for quant in Q8_0 Q4_K_M; do
  out="models/llama-3.1-8b-$(echo "$quant" | tr '[:upper:]' '[:lower:]').gguf"
  [ -f "$out" ] && { echo "skip $out (exists)"; continue; }
  "$LLAMA_CPP_DIR/build/bin/llama-quantize" "$f16" "$out" "$quant"
done
