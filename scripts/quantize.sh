#!/usr/bin/env bash
# Last edited: 2026-10-11 12:00
# Make 8-bit and 4-bit copies of the 16-bit model (group size 64).
set -euo pipefail
cd "$(dirname "$0")/.."
MODEL="${MODEL:-meta-llama/Llama-3.1-8B-Instruct}"
for bits in 8 4; do
  out="models/llama-3.1-8b-q${bits}"
  [ -d "$out" ] && { echo "skip $out (exists)"; continue; }
  uv run mlx_lm.convert --hf-path "$MODEL" --mlx-path "$out" -q --q-bits "$bits" --q-group-size 64
done
