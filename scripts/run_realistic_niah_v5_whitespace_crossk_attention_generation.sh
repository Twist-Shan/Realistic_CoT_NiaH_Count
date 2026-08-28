#!/usr/bin/env bash
set -euo pipefail

ROOT=/home/ubuntu/same_site_progress_20260827
PYTHON=/lambda/nfs/CoT-Native-thinking-v5/venv/bin/python
CACHE=/lambda/nfs/CoT-Native-thinking-v5/cache/huggingface
INPUT="$ROOT/work/input/n10/discovery_rows_first_pass_noindex_v5.jsonl"
OUTPUT="$ROOT/work/output/n10_natural_crossk_attention_generation_v1/whitespace_tail0_l16_w4"
SEEDS=(1293 1359 1384 1539 2322)
mkdir -p "$OUTPUT"
cd "$ROOT"

for DONOR in 4 6 8; do
  for DIRECTION in forward_skip backward_rewind; do
    if [[ "$DIRECTION" == forward_skip ]]; then
      RECEIVER=$((DONOR - 1))
    else
      RECEIVER=$((DONOR + 1))
    fi
    CUDA_VISIBLE_DEVICES=0 "$PYTHON" \
      scripts/run_realistic_niah_v5_natural_aligned_progress_transplant.py \
      --model Qwen3-8B \
      --cache-dir "$CACHE" \
      --device-map auto \
      --generations "$INPUT" \
      --gold-count 10 \
      --receiver-occurrence "$RECEIVER" \
      --donor-occurrence "$DONOR" \
      --tail-offset 0 \
      --patch-width 4 \
      --layers 16 \
      --conditions receiver_self native_donor donor_to_receiver \
      --generation-conditions receiver_self donor_to_receiver \
      --max-new-tokens 96 \
      --run-attention \
      --targeted-selection configs/realistic_niah_v5_qwen_shared_k128_targeted_selection_frozen.json \
      --targeted-routing configs/realistic_niah_v5_qwen_shared_k128_causal_routes_frozen.json \
      --seeds "${SEEDS[@]}" \
      --output "$OUTPUT/${DIRECTION}_k${DONOR}" \
      >"$OUTPUT/${DIRECTION}_k${DONOR}.log" 2>&1
  done
done
