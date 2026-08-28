#!/usr/bin/env bash
set -euo pipefail

ROOT=/home/ubuntu/same_site_progress_20260827
PYTHON=/lambda/nfs/CoT-Native-thinking-v5/venv/bin/python
CACHE=/lambda/nfs/CoT-Native-thinking-v5/cache/huggingface
COHORT_ROOT="$ROOT/work/indexed_progress_control_20260827/cohorts"
RUN_ROOT="$ROOT/work/indexed_progress_control_20260827/runs/confirmation_crossk_v1"
FREEZE_MANIFEST="$ROOT/work/indexed_progress_control_20260827/confirmation_freeze_manifest.json"

QWEN_SEEDS=(1259 1262 1304 1312 1323 1327 1328 1331 1346 1363)
GEMMA_SEEDS=(1630 1655 1727 1758 1816 1896 1926 1984 2048 2072)

selected_layer() {
  "$PYTHON" -c \
    'import json,sys; p=json.load(open(sys.argv[1])); assert p["status"]=="FROZEN_BEFORE_CONFIRMATION" and p["confirmation_results_observed"] is False; print(p["active_confirmation_layers"][sys.argv[2]])' \
    "$FREEZE_MANIFEST" "$1"
}

run_model() {
  local gpu=$1
  local model=$2
  local input=$3
  local selection=$4
  local routing=$5
  shift 5
  local seeds=("$@")
  local layer
  layer=$(selected_layer "$model")
  local model_root="$RUN_ROOT/$model/confirmation10/item_span"
  mkdir -p "$model_root"

  for donor in 4 6 8; do
    for direction in forward_skip backward_rewind; do
      local receiver=$((donor - 1))
      if [[ "$direction" == "backward_rewind" ]]; then
        receiver=$((donor + 1))
      fi
      local output="$model_root/${direction}_k${donor}"
      CUDA_VISIBLE_DEVICES="$gpu" "$PYTHON" \
        scripts/run_realistic_niah_v5_natural_aligned_progress_transplant.py \
        --model "$model" \
        --cache-dir "$CACHE" \
        --device-map auto \
        --attention-backend sdpa \
        --generations "$input" \
        --cohort-mode indexed_positive_control \
        --gold-count 10 \
        --receiver-occurrence "$receiver" \
        --donor-occurrence "$donor" \
        --tail-offset 0 \
        --patch-scope item_span \
        --layers "$layer" \
        --conditions receiver_self native_donor donor_to_receiver \
        --generation-conditions receiver_self donor_to_receiver \
        --max-new-tokens 96 \
        --run-attention \
        --targeted-selection "$selection" \
        --targeted-routing "$routing" \
        --seeds "${seeds[@]}" \
        --output "$output" \
        >"$model_root/${direction}_k${donor}.log" 2>&1
    done
  done

  "$PYTHON" scripts/analyze_realistic_niah_v5_natural_patch_scope_frozen.py \
    "$RUN_ROOT/$model" \
    --output "$RUN_ROOT/$model/frozen_scope_analysis.json" \
    >"$RUN_ROOT/$model/frozen_scope_analysis.log" 2>&1
  printf 'PASS\n' >"$RUN_ROOT/$model/COMPLETE"
}

mkdir -p "$RUN_ROOT/Qwen3-8B" "$RUN_ROOT/Gemma4-E4B"
cd "$ROOT"

run_model \
  0 \
  Qwen3-8B \
  "$COHORT_ROOT/Qwen3-8B.jsonl" \
  configs/realistic_niah_v5_qwen_shared_k128_targeted_selection_frozen.json \
  configs/realistic_niah_v5_qwen_shared_k128_causal_routes_frozen.json \
  "${QWEN_SEEDS[@]}" &
QWEN_PID=$!

run_model \
  1 \
  Gemma4-E4B \
  "$COHORT_ROOT/Gemma4-E4B.jsonl" \
  configs/realistic_niah_v5_gemma_shared_k6_targeted_selection_frozen.json \
  configs/realistic_niah_v5_gemma_shared_k6_causal_routes_frozen.json \
  "${GEMMA_SEEDS[@]}" &
GEMMA_PID=$!

wait "$QWEN_PID"
wait "$GEMMA_PID"
printf 'PASS\n' >"$RUN_ROOT/COMPLETE"
