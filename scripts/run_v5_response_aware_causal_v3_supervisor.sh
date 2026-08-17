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
SUP_GEN=$MODEL_RUN/answer_query_extension_v3/supplement/accepted_generations.jsonl
OLD_PRE_CITY=$MODEL_RUN/attention/pre_city_token/attention.csv
ROOT=$MODEL_RUN/corrected_causal_chain_v3_response_aware
RETRIEVAL=$ROOT/targeted_retrieval_by_response_type
ANSWER=$ROOT/answer_broad_aggregation
CACHE=$FS/cache/huggingface
LOG=$ROOT/logs

mkdir -p "$LOG" "$RETRIEVAL" "$ANSWER"
cd "$REPO"
STATUS=$LOG/supervisor.status
printf 'running\n' > "$STATUS"
printf '%s\n' "$$" > "$LOG/supervisor.pid"
trap 'code=$?; if [[ $code -ne 0 ]]; then printf "failed:%s\n" "$code" > "$STATUS"; fi' EXIT

run_cpu() {
  local name=$1
  shift
  local done=$LOG/$name.done
  if [[ -f "$done" ]]; then
    echo "[$(date -Is)] REUSE $name" | tee -a "$LOG/supervisor.log"
    return
  fi
  echo "[$(date -Is)] START $name" | tee -a "$LOG/supervisor.log"
  "$PY" "$@" > "$LOG/$name.log" 2>&1
  printf 'complete\n' > "$done"
  echo "[$(date -Is)] DONE $name" | tee -a "$LOG/supervisor.log"
}

run_gpu() {
  local name=$1
  shift
  local done=$LOG/$name.done
  if [[ -f "$done" ]]; then
    echo "[$(date -Is)] REUSE $name" | tee -a "$LOG/supervisor.log"
    return
  fi
  echo "[$(date -Is)] START $name gpu=$GPU" | tee -a "$LOG/supervisor.log"
  CUDA_VISIBLE_DEVICES="$GPU" "$PY" "$@" > "$LOG/$name.log" 2>&1
  printf 'complete\n' > "$done"
  echo "[$(date -Is)] DONE $name" | tee -a "$LOG/supervisor.log"
}

# The registered response-reference token is exactly the old pre_city_d1
# position.  Reuse its attention forward only after the identity audit, attach
# the model-specific parser class, and select a separate bank per response type.
run_cpu response_reference_attention \
  scripts/run_realistic_niah_v5.py response-reference-attention \
  --model "$MODEL" --generations "$GEN" \
  --pre-city-attention "$OLD_PRE_CITY" \
  --output "$RETRIEVAL/attention.csv"

run_cpu response_reference_plan \
  scripts/run_realistic_niah_v5.py response-reference-causal-plan \
  --attention "$RETRIEVAL/attention.csv" --output "$RETRIEVAL/plan"

run_gpu response_reference_ablation_primary \
  scripts/run_realistic_niah_v5.py causal-response-reference-heads \
  --model "$MODEL" --generations "$GEN" \
  --plan "$RETRIEVAL/plan/causal_plan.csv" \
  --bank-scopes unified_consensus \
  --cohort one_to_one_correct --split confirmation \
  --output "$RETRIEVAL/trials_primary_confirmation.jsonl" \
  --cache-dir "$CACHE" --device-map auto --torch-dtype bfloat16 \
  --attention-backend sdpa

run_gpu response_reference_ablation_supplement \
  scripts/run_realistic_niah_v5.py causal-response-reference-heads \
  --model "$MODEL" --generations "$SUP_GEN" \
  --plan "$RETRIEVAL/plan/causal_plan.csv" \
  --bank-scopes unified_consensus \
  --cohort one_to_one_correct --split confirmation \
  --output "$RETRIEVAL/trials_supplement_n10_confirmation.jsonl" \
  --cache-dir "$CACHE" --device-map auto --torch-dtype bfloat16 \
  --attention-backend sdpa

run_cpu response_reference_analysis \
  scripts/analyze_v5_next_needle_ablation.py \
  --trials "$RETRIEVAL/trials_primary_confirmation.jsonl" \
  "$RETRIEVAL/trials_supplement_n10_confirmation.jsonl" \
  --output-dir "$RETRIEVAL/analysis"

# Native answer aggregation now follows the non-thinking broad-primary
# definition independently for exact prompt-record spans and registered trace
# item spans.  The per-span vectors were not present in V2, so these two
# attention captures must be recomputed.
run_gpu answer_broad_attention_primary \
  scripts/run_realistic_niah_v5.py attention-answer-query \
  --model "$MODEL" --generations "$GEN" \
  --output "$ANSWER/attention_primary.csv" \
  --site-id answer_query_v3 --cohort one_to_one --split all \
  --cache-dir "$CACHE" --device-map auto --torch-dtype bfloat16 \
  --attention-backend sdpa

run_gpu answer_broad_attention_supplement \
  scripts/run_realistic_niah_v5.py attention-answer-query \
  --model "$MODEL" --generations "$SUP_GEN" \
  --output "$ANSWER/attention_supplement_n10.csv" \
  --site-id answer_query_v3 --cohort one_to_one --split all \
  --cache-dir "$CACHE" --device-map auto --torch-dtype bfloat16 \
  --attention-backend sdpa

run_cpu answer_broad_plan \
  scripts/run_realistic_niah_v5.py answer-query-causal-plan \
  --attention "$ANSWER/attention_primary.csv" --output "$ANSWER/plan"

run_gpu answer_broad_ablation_primary \
  scripts/run_realistic_niah_v5.py causal-answer-query-heads \
  --model "$MODEL" --generations "$GEN" \
  --plan "$ANSWER/plan/answer_query_causal_plan.csv" \
  --site-id answer_query_v3 --cohort one_to_one_correct --split confirmation \
  --output "$ANSWER/trials_primary_confirmation.jsonl" \
  --cache-dir "$CACHE" --device-map auto --torch-dtype bfloat16 \
  --attention-backend sdpa

run_gpu answer_broad_ablation_supplement \
  scripts/run_realistic_niah_v5.py causal-answer-query-heads \
  --model "$MODEL" --generations "$SUP_GEN" \
  --plan "$ANSWER/plan/answer_query_causal_plan.csv" \
  --site-id answer_query_v3 --cohort one_to_one_correct --split confirmation \
  --output "$ANSWER/trials_supplement_n10_confirmation.jsonl" \
  --cache-dir "$CACHE" --device-map auto --torch-dtype bfloat16 \
  --attention-backend sdpa

run_cpu answer_broad_analysis \
  scripts/analyze_v5_greedy_head_damage.py \
  --trials "$ANSWER/trials_primary_confirmation.jsonl" \
  "$ANSWER/trials_supplement_n10_confirmation.jsonl" \
  --output-dir "$ANSWER/analysis"

printf 'complete\n' > "$STATUS"
echo "[$(date -Is)] COMPLETE $MODEL" | tee -a "$LOG/supervisor.log"
