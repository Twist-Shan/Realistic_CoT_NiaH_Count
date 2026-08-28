#!/usr/bin/env bash
set -euo pipefail

MODEL=${1:?usage: $0 MODEL GPU_INDEX}
GPU_INDEX=${2:?usage: $0 MODEL GPU_INDEX}
ROOT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$ROOT_DIR"

PYTHON="$ROOT_DIR/.venv/bin/python"
STIMULI="$ROOT_DIR/work/v5_native_count_stream/frozen_prompt_bullet_rate_pilot_100seeds_20260823_v1/stimuli/stimuli.jsonl"
OUTPUT="$ROOT_DIR/work/v5_native_count_stream/frozen_prompt_bullet_rate_pilot_100seeds_20260823_v1/$MODEL"
LOG="$OUTPUT/logs/supervisor.log"
LOCK="$OUTPUT/locks/supervisor.lock"
mkdir -p "$OUTPUT/logs" "$OUTPUT/locks"
exec 9>"$LOCK"
flock -n 9 || { echo "another $MODEL frozen-prompt bullet pilot owns the lock" >&2; exit 3; }

SEEDS=()
for seed in $(seq 3000 3099); do SEEDS+=("$seed"); done

CUDA_VISIBLE_DEVICES="$GPU_INDEX" "$PYTHON" \
  scripts/run_realistic_niah_v5_frozen_prompt_bullet_pilot.py \
  --model "$MODEL" \
  --cache-dir "$ROOT_DIR/work/hf_cache" \
  --device-map auto \
  --torch-dtype bfloat16 \
  --attention-backend sdpa \
  --stimuli "$STIMULI" \
  --expected-seeds "${SEEDS[@]}" \
  --counts 10 9 \
  --resume \
  --output "$OUTPUT" 2>&1 | tee -a "$LOG"
