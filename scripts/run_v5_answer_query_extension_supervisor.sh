#!/usr/bin/env bash
set -euo pipefail

MODEL=${1:?model label required}
GPU=${2:?gpu index required}
FS=/home/ubuntu/CoT-Native-thinking-v5
REPO=$FS/code/Realistic_CoT_NiaH_Count
PY=$FS/venv/bin/python
RUN=$FS/runs/v5_native_thinking_representation
MODEL_RUN=$RUN/$MODEL
EXT=$MODEL_RUN/answer_query_extension
SUP_ROOT=$RUN/one_to_one_supplement
GEN=$MODEL_RUN/generations.jsonl
SUP_GEN=$EXT/supplement/accepted_generations.jsonl
LOG=$EXT/logs
CACHE=$FS/cache/huggingface

mkdir -p "$LOG" "$EXT/supplement"
cd "$REPO"

STATUS=$LOG/supervisor.status
printf 'running\n' > "$STATUS"
printf '%s\n' "$$" > "$LOG/supervisor.pid"
trap 'code=$?; if [[ $code -ne 0 ]]; then printf "failed:%s\n" "$code" > "$STATUS"; fi' EXIT

run_gpu() {
  local name=$1
  shift
  echo "[$(date -Is)] START $name" | tee -a "$LOG/supervisor.log"
  CUDA_VISIBLE_DEVICES="$GPU" "$PY" "$@" > "$LOG/$name.log" 2>&1
  echo "[$(date -Is)] DONE $name" | tee -a "$LOG/supervisor.log"
}

run_cpu() {
  local name=$1
  shift
  echo "[$(date -Is)] START $name" | tee -a "$LOG/supervisor.log"
  "$PY" "$@" > "$LOG/$name.log" 2>&1
  echo "[$(date -Is)] DONE $name" | tee -a "$LOG/supervisor.log"
}

run_cpu materialize_supplement \
  scripts/materialize_v5_one_to_one_accepted_generations.py \
  --supplement-root "$SUP_ROOT" --model "$MODEL" --output "$SUP_GEN"

run_gpu capture_primary \
  scripts/run_realistic_niah_v5.py capture \
  --model "$MODEL" --generations "$GEN" \
  --output "$EXT/representation/capture_primary" \
  --site-ids answer_query_v2 --cache-dir "$CACHE" --device-map auto \
  --torch-dtype bfloat16 --attention-backend sdpa

run_gpu capture_supplement \
  scripts/run_realistic_niah_v5.py capture \
  --model "$MODEL" --generations "$SUP_GEN" \
  --output "$EXT/representation/capture_supplement_n10" \
  --site-ids answer_query_v2 --allow-unregistered \
  --cache-dir "$CACHE" --device-map auto --torch-dtype bfloat16 \
  --attention-backend sdpa

run_cpu representation_primary \
  scripts/run_realistic_niah_v5.py representation \
  --capture-index "$EXT/representation/capture_primary/capture_index.jsonl" \
  --output "$EXT/representation/analysis_primary"

run_gpu attention_primary \
  scripts/run_realistic_niah_v5.py attention-answer-query \
  --model "$MODEL" --generations "$GEN" \
  --output "$EXT/head_ablation/attention_primary.csv" \
  --site-id answer_query_v2 --cohort one_to_one --split all \
  --cache-dir "$CACHE" --device-map auto --torch-dtype bfloat16 \
  --attention-backend sdpa

run_gpu attention_supplement \
  scripts/run_realistic_niah_v5.py attention-answer-query \
  --model "$MODEL" --generations "$SUP_GEN" \
  --output "$EXT/head_ablation/attention_supplement_n10.csv" \
  --site-id answer_query_v2 --cohort one_to_one --split all \
  --cache-dir "$CACHE" --device-map auto --torch-dtype bfloat16 \
  --attention-backend sdpa

run_cpu answer_head_plan \
  scripts/run_realistic_niah_v5.py answer-query-causal-plan \
  --attention "$EXT/head_ablation/attention_primary.csv" \
  --output "$EXT/head_ablation/plan"

run_gpu head_ablation_primary \
  scripts/run_realistic_niah_v5.py causal-answer-query-heads \
  --model "$MODEL" --generations "$GEN" \
  --plan "$EXT/head_ablation/plan/answer_query_causal_plan.csv" \
  --output "$EXT/head_ablation/trials_primary_confirmation.jsonl" \
  --site-id answer_query_v2 --cohort one_to_one --split confirmation \
  --cache-dir "$CACHE" --device-map auto --torch-dtype bfloat16 \
  --attention-backend sdpa

run_gpu head_ablation_supplement \
  scripts/run_realistic_niah_v5.py causal-answer-query-heads \
  --model "$MODEL" --generations "$SUP_GEN" \
  --plan "$EXT/head_ablation/plan/answer_query_causal_plan.csv" \
  --output "$EXT/head_ablation/trials_supplement_n10_confirmation.jsonl" \
  --site-id answer_query_v2 --cohort one_to_one --split confirmation \
  --cache-dir "$CACHE" --device-map auto --torch-dtype bfloat16 \
  --attention-backend sdpa

run_cpu answer_execution_plan \
  scripts/build_v5_answer_execution_plan.py \
  --capture-index "$EXT/representation/capture_primary/capture_index.jsonl" \
  --generations "$GEN" --model "$MODEL" \
  --output-dir "$EXT/answer_execution/plan" \
  --site-kind answer_query_v2 --site-id answer_query_v2 --rank 3

run_gpu answer_execution \
  scripts/run_realistic_niah_v5.py causal-patch \
  --model "$MODEL" --generations "$GEN" \
  --pairs "$EXT/answer_execution/plan/${MODEL}__answer_execution_pairs.jsonl" \
  --basis "$EXT/answer_execution/plan/${MODEL}__answer_query_v2_basis.npz" \
  --output "$EXT/answer_execution/trials_confirmation.jsonl" \
  --layer "$("$PY" -c "import json; print(json.load(open('$EXT/answer_execution/plan/${MODEL}__answer_execution_layer_selection.json'))['selected_layer'])")" \
  --receiver-site-id answer_query_v2 --donor-site-id answer_query_v2 \
  --max-new-tokens 16 --restartable \
  --cache-dir "$CACHE" --device-map auto --torch-dtype bfloat16 \
  --attention-backend sdpa

printf 'complete\n' > "$STATUS"
echo "[$(date -Is)] COMPLETE $MODEL" | tee -a "$LOG/supervisor.log"
