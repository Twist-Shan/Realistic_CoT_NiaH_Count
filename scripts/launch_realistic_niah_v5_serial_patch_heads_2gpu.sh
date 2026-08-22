#!/usr/bin/env bash
set -euo pipefail

CODE_ROOT="${CODE_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "$CODE_ROOT"

nohup env CUDA_VISIBLE_DEVICES=0 \
  bash "$CODE_ROOT/scripts/supervise_realistic_niah_v5_serial_patch_heads.sh" \
  Qwen3-8B >"$CODE_ROOT/qwen-serial-patch-heads-launch.log" 2>&1 &
QWEN_PID=$!

nohup env CUDA_VISIBLE_DEVICES=1 \
  bash "$CODE_ROOT/scripts/supervise_realistic_niah_v5_serial_patch_heads.sh" \
  Gemma4-E4B >"$CODE_ROOT/gemma-serial-patch-heads-launch.log" 2>&1 &
GEMMA_PID=$!

printf 'Qwen3-8B supervisor pid=%s gpu=0\n' "$QWEN_PID"
printf 'Gemma4-E4B supervisor pid=%s gpu=1\n' "$GEMMA_PID"
