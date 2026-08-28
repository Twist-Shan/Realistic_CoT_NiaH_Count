#!/usr/bin/env bash
set -euo pipefail

ROOT=/home/ubuntu/same_site_progress_20260827
PYTHON=/lambda/nfs/CoT-Native-thinking-v5/venv/bin/python
CACHE=/lambda/nfs/CoT-Native-thinking-v5/cache/huggingface
INPUT="$ROOT/work/input/n10/discovery_rows_first_pass_noindex_v5.jsonl"
OUTPUT="$ROOT/work/output/n10_patch_scope_layer_sweep_v2"
SEEDS=(1267 1290 1293 1359 1384 1506 1539 1621 1750 1771 1791 1810 1893 2056 2128 2254 2295 2298 2302 2322)
LAYERS=(0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 32 33 34 35)
mkdir -p "$OUTPUT"
cd "$ROOT"

run_cell() {
  local gpu=$1
  local scope=$2
  local direction=$3
  local receiver=$4
  shift 4
  local output="$OUTPUT/$scope/${direction}_k6"
  mkdir -p "$output"
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
    "$@" \
    --layers "${LAYERS[@]}" \
    --conditions donor_to_receiver \
    --seeds "${SEEDS[@]}" \
    --output "$output" \
    >"$OUTPUT/$scope/${direction}_k6.log" 2>&1
}

run_baseline() {
  local gpu=$1
  local direction=$2
  local receiver=$3
  local output="$OUTPUT/baseline/${direction}_k6"
  mkdir -p "$output"
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
    --patch-scope fixed_suffix \
    --patch-width 1 \
    --layers 0 \
    --conditions receiver_self native_donor \
    --seeds "${SEEDS[@]}" \
    --output "$output" \
    >"$OUTPUT/baseline/${direction}_k6.log" 2>&1
}

worker_zero() {
  run_baseline 0 forward_skip 5
  run_baseline 0 backward_rewind 7
  run_cell 0 item_end_w1 forward_skip 5 --patch-scope fixed_suffix --patch-width 1
  run_cell 0 item_end_w1 backward_rewind 7 --patch-scope fixed_suffix --patch-width 1
  run_cell 0 event_tail_w4 forward_skip 5 --patch-scope fixed_suffix --patch-width 4
}

worker_one() {
  run_cell 1 item_span forward_skip 5 --patch-scope item_span
  run_cell 1 item_span backward_rewind 7 --patch-scope item_span
  run_cell 1 event_tail_w4 backward_rewind 7 --patch-scope fixed_suffix --patch-width 4
}

worker_zero &
PID_ZERO=$!
worker_one &
PID_ONE=$!
wait "$PID_ZERO"
wait "$PID_ONE"

"$PYTHON" scripts/analyze_realistic_niah_v5_natural_patch_scope_layer_sweep.py \
  "$OUTPUT" \
  --output "$OUTPUT/layer_sweep_analysis.json" \
  >"$OUTPUT/layer_sweep_analysis.log" 2>&1
printf 'PASS\n' >"$OUTPUT/COMPLETE"
