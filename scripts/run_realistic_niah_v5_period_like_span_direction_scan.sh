#!/usr/bin/env bash
set -euo pipefail

ROOT=/home/ubuntu/same_site_progress_20260827
PYTHON=/lambda/nfs/CoT-Native-thinking-v5/venv/bin/python
CACHE=/lambda/nfs/CoT-Native-thinking-v5/cache/huggingface
INPUT="$ROOT/work/input/n10/discovery_rows_first_pass_noindex_v5.jsonl"
OUTPUT="$ROOT/work/output/n10_natural_span_direction_localization_v1/period_like_l16_k6"
SEEDS=(1290 1750 1771 1810 1893)
mkdir -p "$OUTPUT"
cd "$ROOT"

for RECEIVER in 5 7; do
  if [[ "$RECEIVER" == 5 ]]; then
    DIRECTION=forward_skip
  else
    DIRECTION=backward_rewind
  fi
  for WIDTH in 1 2 4; do
    CUDA_VISIBLE_DEVICES=1 "$PYTHON" \
      scripts/run_realistic_niah_v5_natural_aligned_progress_transplant.py \
      --model Qwen3-8B \
      --cache-dir "$CACHE" \
      --device-map auto \
      --generations "$INPUT" \
      --gold-count 10 \
      --receiver-occurrence "$RECEIVER" \
      --donor-occurrence 6 \
      --tail-window 12 \
      --site-policy period_preferred \
      --patch-width "$WIDTH" \
      --layers 16 \
      --conditions receiver_self native_donor donor_to_receiver \
      --seeds "${SEEDS[@]}" \
      --output "$OUTPUT/${DIRECTION}_w${WIDTH}" \
      >"$OUTPUT/${DIRECTION}_w${WIDTH}.log" 2>&1
  done
done
