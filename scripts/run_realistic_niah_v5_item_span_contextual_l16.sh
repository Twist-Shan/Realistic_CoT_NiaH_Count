#!/usr/bin/env bash
set -euo pipefail

ROOT=/home/ubuntu/same_site_progress_20260827
PYTHON=/lambda/nfs/CoT-Native-thinking-v5/venv/bin/python
CACHE=/lambda/nfs/CoT-Native-thinking-v5/cache/huggingface
INPUT="$ROOT/work/input/n10/confirmation_rows_first_pass_noindex_v5.jsonl"
OUTPUT="$ROOT/work/output/n10_item_span_contextual_l16_v1/confirmation10/item_span_l16"
TARGET_SELECTION=configs/realistic_niah_v5_qwen_shared_k128_targeted_selection_frozen.json
TARGET_ROUTING=configs/realistic_niah_v5_qwen_shared_k128_causal_routes_frozen.json
SEEDS=(1307 1364 1553 1598 1688 1805 1979 1982 2009 2024)
mkdir -p "$OUTPUT"
cd "$ROOT"

run_direction() {
  local gpu=$1
  local direction=$2
  local receiver
  if [[ "$direction" == forward_skip ]]; then
    receiver=5
  else
    receiver=7
  fi
  local cell="$OUTPUT/${direction}_k6"
  mkdir -p "$cell"
  CUDA_VISIBLE_DEVICES="$gpu" "$PYTHON" \
    scripts/run_realistic_niah_v5_natural_aligned_progress_transplant.py \
    --model Qwen3-8B \
    --cache-dir "$CACHE" \
    --device-map auto \
    --generations "$INPUT" \
    --gold-count 10 \
    --receiver-occurrence "$receiver" \
    --donor-occurrence 6 \
    --tail-offset 0 \
    --patch-scope item_span \
    --layers 16 \
    --conditions receiver_self native_donor donor_to_receiver \
    --generation-conditions receiver_self donor_to_receiver \
    --max-new-tokens 96 \
    --run-attention \
    --targeted-selection "$TARGET_SELECTION" \
    --targeted-routing "$TARGET_ROUTING" \
    --seeds "${SEEDS[@]}" \
    --output "$cell" \
    >"$OUTPUT/${direction}_k6.log" 2>&1
}

run_direction 0 forward_skip &
PID_FORWARD=$!
run_direction 1 backward_rewind &
PID_BACKWARD=$!
wait "$PID_FORWARD"
wait "$PID_BACKWARD"

"$PYTHON" scripts/analyze_realistic_niah_v5_natural_patch_scope_frozen.py \
  "$ROOT/work/output/n10_item_span_contextual_l16_v1" \
  --output "$ROOT/work/output/n10_item_span_contextual_l16_v1/frozen_scope_analysis.json" \
  >"$ROOT/work/output/n10_item_span_contextual_l16_v1/frozen_scope_analysis.log" 2>&1
printf 'PASS\n' >"$ROOT/work/output/n10_item_span_contextual_l16_v1/COMPLETE"
