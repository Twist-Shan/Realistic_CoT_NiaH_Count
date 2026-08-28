#!/usr/bin/env bash
set -euo pipefail

ROOT=/home/ubuntu/same_site_progress_20260827
PYTHON=/lambda/nfs/CoT-Native-thinking-v5/venv/bin/python
CACHE=/lambda/nfs/CoT-Native-thinking-v5/cache/huggingface
COHORT_ROOT="$ROOT/work/indexed_progress_control_20260827/cohorts"
RUN_ROOT="$ROOT/work/indexed_progress_control_20260827/runs/discovery_layer_sweep_v1"

QWEN_SEEDS=(1242 1252 1255 1257 1269 1280 1284 1289 1290 1293 1295 1299 1300 1320 1321 1330 1339 1340 1351 1355)
GEMMA_SEEDS=(1629 1635 1675 1682 1689 1698 1761 1792 1810 1818 1861 1885 1923 1938 1974 1975 1986 2001 2014 2052)
QWEN_LAYERS=($(seq 0 35))
GEMMA_LAYERS=($(seq 0 41))

run_panel() {
  local gpu=$1
  local model=$2
  local input=$3
  local output_root=$4
  shift 4
  local seeds=("$@")
  local layers=()
  if [[ "$model" == "Qwen3-8B" ]]; then
    layers=("${QWEN_LAYERS[@]}")
  else
    layers=("${GEMMA_LAYERS[@]}")
  fi
  mkdir -p "$output_root/baseline" "$output_root/item_span"

  for direction in forward_skip backward_rewind; do
    local receiver=5
    if [[ "$direction" == "backward_rewind" ]]; then
      receiver=7
    fi
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
      --donor-occurrence 6 \
      --tail-offset 0 \
      --patch-scope fixed_suffix \
      --patch-width 1 \
      --layers 0 \
      --conditions receiver_self native_donor \
      --seeds "${seeds[@]}" \
      --output "$output_root/baseline/${direction}_k6" \
      >"$output_root/baseline/${direction}_k6.log" 2>&1

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
      --donor-occurrence 6 \
      --tail-offset 0 \
      --patch-scope item_span \
      --layers "${layers[@]}" \
      --conditions donor_to_receiver \
      --seeds "${seeds[@]}" \
      --output "$output_root/item_span/${direction}_k6" \
      >"$output_root/item_span/${direction}_k6.log" 2>&1
  done

  "$PYTHON" scripts/analyze_realistic_niah_v5_natural_patch_scope_layer_sweep.py \
    "$output_root" \
    --output "$output_root/layer_sweep_analysis.json" \
    >"$output_root/layer_sweep_analysis.log" 2>&1
  printf 'PASS\n' >"$output_root/COMPLETE"
}

mkdir -p "$RUN_ROOT/Qwen3-8B" "$RUN_ROOT/Gemma4-E4B"
cd "$ROOT"

run_panel \
  0 \
  Qwen3-8B \
  "$COHORT_ROOT/Qwen3-8B.jsonl" \
  "$RUN_ROOT/Qwen3-8B" \
  "${QWEN_SEEDS[@]}" &
QWEN_PID=$!

run_panel \
  1 \
  Gemma4-E4B \
  "$COHORT_ROOT/Gemma4-E4B.jsonl" \
  "$RUN_ROOT/Gemma4-E4B" \
  "${GEMMA_SEEDS[@]}" &
GEMMA_PID=$!

wait "$QWEN_PID"
wait "$GEMMA_PID"
printf 'PASS\n' >"$RUN_ROOT/COMPLETE"
