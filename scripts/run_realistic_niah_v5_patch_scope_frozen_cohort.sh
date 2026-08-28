#!/usr/bin/env bash
set -euo pipefail

ROOT=/home/ubuntu/same_site_progress_20260827
PYTHON=/lambda/nfs/CoT-Native-thinking-v5/venv/bin/python
CACHE=/lambda/nfs/CoT-Native-thinking-v5/cache/huggingface
SELECTION="$ROOT/work/output/n10_patch_scope_layer_sweep_v2/layer_sweep_analysis.json"
OUTPUT="$ROOT/work/output/n10_patch_scope_frozen_v2"
TARGET_SELECTION=configs/realistic_niah_v5_qwen_shared_k128_targeted_selection_frozen.json
TARGET_ROUTING=configs/realistic_niah_v5_qwen_shared_k128_causal_routes_frozen.json
DISCOVERY_INPUT="$ROOT/work/input/n10/discovery_rows_first_pass_noindex_v5.jsonl"
CONFIRMATION_INPUT="$ROOT/work/input/n10/confirmation_rows_first_pass_noindex_v5.jsonl"
DISCOVERY_SEEDS=(1267 1290 1293 1359 1384 1506 1539 1621 1750 1771 1791 1810 1893 2056 2128 2254 2295 2298 2302 2322)
CONFIRMATION_SEEDS=(1307 1364 1553 1598 1688 1805 1979 1982 2009 2024)
mkdir -p "$OUTPUT"
cd "$ROOT"

selected_layer() {
  "$PYTHON" -c 'import json,sys; p=json.load(open(sys.argv[1])); s=sys.argv[2]; print(next(x["selected_layer"] for x in p["scopes"] if x["scope"]==s))' "$SELECTION" "$1"
}

LAYER_ITEM_END=$(selected_layer item_end_w1)
LAYER_EVENT_TAIL=$(selected_layer event_tail_w4)
LAYER_ITEM_SPAN=$(selected_layer item_span)

run_cell() {
  local gpu=$1
  local split=$2
  local input=$3
  local scope=$4
  local direction=$5
  local donor=$6
  local receiver=$7
  local layer=$8
  shift 8
  local output="$OUTPUT/$split/$scope/${direction}_k${donor}"
  mkdir -p "$output"
  local -a seeds
  if [[ "$split" == discovery20 ]]; then
    seeds=("${DISCOVERY_SEEDS[@]}")
  else
    seeds=("${CONFIRMATION_SEEDS[@]}")
  fi
  CUDA_VISIBLE_DEVICES="$gpu" "$PYTHON" \
    scripts/run_realistic_niah_v5_natural_aligned_progress_transplant.py \
    --model Qwen3-8B \
    --cache-dir "$CACHE" \
    --device-map auto \
    --generations "$input" \
    --gold-count 10 \
    --receiver-occurrence "$receiver" \
    --donor-occurrence "$donor" \
    --tail-offset 0 \
    "$@" \
    --layers "$layer" \
    --conditions receiver_self native_donor donor_to_receiver \
    --generation-conditions receiver_self donor_to_receiver \
    --max-new-tokens 96 \
    --run-attention \
    --targeted-selection "$TARGET_SELECTION" \
    --targeted-routing "$TARGET_ROUTING" \
    --seeds "${seeds[@]}" \
    --output "$output" \
    >"$OUTPUT/$split/$scope/${direction}_k${donor}.log" 2>&1
}

run_direction() {
  local gpu=$1
  local direction=$2
  local donor receiver
  local split=confirmation10
  local input="$CONFIRMATION_INPUT"
  for donor in 4 6 8; do
    if [[ "$direction" == forward_skip ]]; then
      receiver=$((donor - 1))
    else
      receiver=$((donor + 1))
    fi
    run_cell "$gpu" "$split" "$input" item_end_w1 "$direction" "$donor" "$receiver" "$LAYER_ITEM_END" --patch-scope fixed_suffix --patch-width 1
    run_cell "$gpu" "$split" "$input" event_tail_w4 "$direction" "$donor" "$receiver" "$LAYER_EVENT_TAIL" --patch-scope fixed_suffix --patch-width 4
    run_cell "$gpu" "$split" "$input" item_span "$direction" "$donor" "$receiver" "$LAYER_ITEM_SPAN" --patch-scope item_span
  done
}

run_direction 0 forward_skip &
PID_FORWARD=$!
run_direction 1 backward_rewind &
PID_BACKWARD=$!
wait "$PID_FORWARD"
wait "$PID_BACKWARD"

"$PYTHON" scripts/analyze_realistic_niah_v5_natural_patch_scope_frozen.py \
  "$OUTPUT" \
  --output "$OUTPUT/frozen_scope_analysis.json" \
  >"$OUTPUT/frozen_scope_analysis.log" 2>&1
printf 'PASS\n' >"$OUTPUT/COMPLETE"
