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
V3=$MODEL_RUN/corrected_causal_chain_v3_response_aware
ROOT=$V3/targeted_retrieval_trace_pre_needle_d1
CACHE=$FS/cache/huggingface
LOG=$ROOT/logs

mkdir -p "$LOG" "$ROOT"
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

# The dedicated parser registers every response occurrence k, finds the start
# of its copied citation (canonical prompt-record prefix when present, exact
# city fallback otherwise), takes the literal baseline token immediately to
# its left, and resolves the k-matched exact prompt needle span by city identity.
# Discovery then freezes one equal-response-type bank at this single position.
run_gpu trace_pre_needle_d1_attention_primary \
  scripts/run_realistic_niah_v5.py attention-targeted-reference \
  --model "$MODEL" --generations "$GEN" \
  --output "$ROOT/attention_primary.csv" --split all \
  --cache-dir "$CACHE" --device-map auto --torch-dtype bfloat16 \
  --attention-backend sdpa

run_cpu trace_pre_needle_d1_plan \
  scripts/run_realistic_niah_v5.py response-reference-causal-plan \
  --attention "$ROOT/attention_primary.csv" --output "$ROOT/plan"

run_gpu trace_pre_needle_d1_ablation_primary \
  scripts/run_realistic_niah_v5.py causal-response-reference-heads \
  --model "$MODEL" --generations "$GEN" \
  --plan "$ROOT/plan/causal_plan.csv" \
  --bank-scopes position_consensus --position-variants pre_reference_d1 \
  --cohort one_to_one_correct --split confirmation \
  --output "$ROOT/trials_primary_confirmation.jsonl" \
  --cache-dir "$CACHE" --device-map auto --torch-dtype bfloat16 \
  --attention-backend sdpa

run_gpu trace_pre_needle_d1_ablation_supplement \
  scripts/run_realistic_niah_v5.py causal-response-reference-heads \
  --model "$MODEL" --generations "$SUP_GEN" \
  --plan "$ROOT/plan/causal_plan.csv" \
  --bank-scopes position_consensus --position-variants pre_reference_d1 \
  --cohort one_to_one_correct --split confirmation \
  --output "$ROOT/trials_supplement_n10_confirmation.jsonl" \
  --cache-dir "$CACHE" --device-map auto --torch-dtype bfloat16 \
  --attention-backend sdpa

run_cpu trace_pre_needle_d1_analysis \
  scripts/analyze_v5_next_needle_ablation.py \
  --trials "$ROOT/trials_primary_confirmation.jsonl" \
  "$ROOT/trials_supplement_n10_confirmation.jsonl" \
  --output-dir "$ROOT/analysis"

printf 'complete\n' > "$STATUS"
echo "[$(date -Is)] COMPLETE $MODEL" | tee -a "$LOG/supervisor.log"
