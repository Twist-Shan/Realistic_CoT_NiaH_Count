#!/usr/bin/env bash
set -euo pipefail

ROOT=/home/ubuntu/same_site_progress_20260827
PYTHON=/lambda/nfs/CoT-Native-thinking-v5/venv/bin/python
CACHE=/lambda/nfs/CoT-Native-thinking-v5/cache/huggingface
INPUT="$ROOT/work/input/n10/discovery_rows_first_pass_noindex_v5.jsonl"
OUTPUT="$ROOT/work/output/n10_natural_site_layer_localization_v1"
LAYERS=(0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 32 33 34 35)

mkdir -p "$OUTPUT"
cd "$ROOT"

CUDA_VISIBLE_DEVICES=0 "$PYTHON" \
  scripts/run_realistic_niah_v5_natural_aligned_progress_transplant.py \
  --model Qwen3-8B \
  --cache-dir "$CACHE" \
  --device-map auto \
  --generations "$INPUT" \
  --gold-count 10 \
  --receiver-occurrence 5 \
  --donor-occurrence 6 \
  --tail-offset 0 \
  --layers "${LAYERS[@]}" \
  --conditions receiver_self native_donor donor_to_receiver \
  --seeds 1293 1359 1384 1539 2322 \
  --output "$OUTPUT/whitespace_tail0_k6" \
  >"$OUTPUT/whitespace_tail0_k6.log" 2>&1 &
PID0=$!

CUDA_VISIBLE_DEVICES=1 "$PYTHON" \
  scripts/run_realistic_niah_v5_natural_aligned_progress_transplant.py \
  --model Qwen3-8B \
  --cache-dir "$CACHE" \
  --device-map auto \
  --generations "$INPUT" \
  --gold-count 10 \
  --receiver-occurrence 5 \
  --donor-occurrence 6 \
  --tail-window 12 \
  --site-policy period_preferred \
  --layers "${LAYERS[@]}" \
  --conditions receiver_self native_donor donor_to_receiver \
  --seeds 1290 1750 1771 1810 1893 \
  --output "$OUTPUT/period_like_k6" \
  >"$OUTPUT/period_like_k6.log" 2>&1 &
PID1=$!

STATUS=0
wait "$PID0" || STATUS=1
wait "$PID1" || STATUS=1
printf 'status=%s gpu0_pid=%s gpu1_pid=%s\n' "$STATUS" "$PID0" "$PID1" >"$OUTPUT/supervisor.status"
exit "$STATUS"
