#!/usr/bin/env bash
set -euo pipefail

MODEL=${1:?model label required}
GPU=${2:?gpu index required}
FS=/home/ubuntu/CoT-Native-thinking-v5
REPO=$FS/code/Realistic_CoT_NiaH_Count
PY=$FS/venv/bin/python
RUN=$FS/runs/v5_native_thinking_representation
MODEL_RUN=$RUN/$MODEL
GEN=$MODEL_RUN/generations.jsonl
EXT=$MODEL_RUN/answer_query_extension_v3
SUP_GEN=$EXT/supplement/accepted_generations.jsonl
CACHE=$FS/cache/huggingface
LOG=$EXT/logs
PLAN=$MODEL_RUN/causal/pre_city_token/plan/causal_plan.csv
GLOBAL=$MODEL_RUN/causal/trace_rollout_damage
WORKER_ID="takeover_gpu${GPU}_$$"

mkdir -p "$LOG" "$GLOBAL/logs"
cd "$REPO"
printf '%s\n' "$$" > "$LOG/takeover_gpu${GPU}.pid"
echo "[$(date -Is)] START takeover worker=$WORKER_ID model=$MODEL" \
  | tee -a "$LOG/takeover_gpu${GPU}.log"

run_shared() {
  local stage=$1
  local variant=$2
  local generations=$3
  local output=$4
  local allow_unregistered=$5
  if [[ -f "$LOG/$stage.done" ]]; then
    echo "[$(date -Is)] SKIP completed $stage" \
      | tee -a "$LOG/takeover_gpu${GPU}.log"
    return
  fi
  local extra=()
  if [[ "$allow_unregistered" == 1 ]]; then
    extra+=(--allow-unregistered)
  fi
  echo "[$(date -Is)] JOIN $stage worker=$WORKER_ID" \
    | tee -a "$LOG/takeover_gpu${GPU}.log"
  CUDA_VISIBLE_DEVICES="$GPU" "$PY" scripts/run_realistic_niah_v5.py \
    causal-pre-city-all-sites \
    --model "$MODEL" --generations "$generations" --plan "$PLAN" \
    --query-variant "$variant" \
    --cohort one_to_one --output "$output" \
    --worker-id "$WORKER_ID" "${extra[@]}" \
    --cache-dir "$CACHE" --device-map auto --torch-dtype bfloat16 \
    --attention-backend sdpa \
    >> "$LOG/takeover_gpu${GPU}.log" 2>&1
}

for variant in pre_city_d1 pre_city_d2 pre_city_anchor; do
  run_shared \
    "all_site_damage_${variant}" \
    "$variant" \
    "$GEN" \
    "$GLOBAL/${variant}/trials_primary_confirmation.jsonl" \
    0
  run_shared \
    "all_site_damage_supplement_${variant}" \
    "$variant" \
    "$SUP_GEN" \
    "$GLOBAL/${variant}/trials_supplement_n10_confirmation.jsonl" \
    1
done

echo "[$(date -Is)] COMPLETE takeover worker=$WORKER_ID model=$MODEL" \
  | tee -a "$LOG/takeover_gpu${GPU}.log"
