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
SUP_ROOT=$RUN/one_to_one_supplement
SUP_GEN=$EXT/supplement/accepted_generations.jsonl
CACHE=$FS/cache/huggingface
LOG=$EXT/logs
E4_PLAN=$MODEL_RUN/causal/pre_city_token/plan/causal_plan.csv
GLOBAL=$MODEL_RUN/causal/trace_rollout_damage

mkdir -p "$LOG" "$EXT/supplement" "$GLOBAL/logs"
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
  echo "[$(date -Is)] START $name" | tee -a "$LOG/supervisor.log"
  CUDA_VISIBLE_DEVICES="$GPU" "$PY" "$@" > "$LOG/$name.log" 2>&1
  printf 'complete\n' > "$done"
  echo "[$(date -Is)] DONE $name" | tee -a "$LOG/supervisor.log"
}

run_cpu materialize_supplement \
  scripts/materialize_v5_one_to_one_accepted_generations.py \
  --supplement-root "$SUP_ROOT" --model "$MODEL" --output "$SUP_GEN"

run_gpu capture_primary \
  scripts/run_realistic_niah_v5.py capture \
  --model "$MODEL" --generations "$GEN" \
  --output "$EXT/representation/capture_primary" \
  --site-ids answer_query_v3 --cache-dir "$CACHE" --device-map auto \
  --torch-dtype bfloat16 --attention-backend sdpa

run_gpu capture_supplement \
  scripts/run_realistic_niah_v5.py capture \
  --model "$MODEL" --generations "$SUP_GEN" \
  --output "$EXT/representation/capture_supplement_n10" \
  --site-ids answer_query_v3 --allow-unregistered \
  --cache-dir "$CACHE" --device-map auto --torch-dtype bfloat16 \
  --attention-backend sdpa

run_cpu merge_capture_indices \
  scripts/merge_v5_answer_query_capture_indices.py \
  --primary "$EXT/representation/capture_primary/capture_index.jsonl" \
  --supplement "$EXT/representation/capture_supplement_n10/capture_index.jsonl" \
  --output "$EXT/representation/capture_combined/capture_index.jsonl"

run_cpu representation_combined \
  scripts/run_realistic_niah_v5.py representation \
  --capture-index "$EXT/representation/capture_combined/capture_index.jsonl" \
  --output "$EXT/representation/analysis_combined"

run_cpu representation_audit \
  scripts/check_v5_representation_output.py \
  "$EXT/representation/analysis_combined"

run_gpu attention_primary \
  scripts/run_realistic_niah_v5.py attention-answer-query \
  --model "$MODEL" --generations "$GEN" \
  --output "$EXT/head_ablation/attention_primary.csv" \
  --site-id answer_query_v3 --cohort one_to_one --split all \
  --cache-dir "$CACHE" --device-map auto --torch-dtype bfloat16 \
  --attention-backend sdpa

run_gpu attention_supplement \
  scripts/run_realistic_niah_v5.py attention-answer-query \
  --model "$MODEL" --generations "$SUP_GEN" \
  --output "$EXT/head_ablation/attention_supplement_n10.csv" \
  --site-id answer_query_v3 --cohort one_to_one --split all \
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
  --site-id answer_query_v3 --cohort one_to_one --split confirmation \
  --cache-dir "$CACHE" --device-map auto --torch-dtype bfloat16 \
  --attention-backend sdpa

run_gpu head_ablation_supplement \
  scripts/run_realistic_niah_v5.py causal-answer-query-heads \
  --model "$MODEL" --generations "$SUP_GEN" \
  --plan "$EXT/head_ablation/plan/answer_query_causal_plan.csv" \
  --output "$EXT/head_ablation/trials_supplement_n10_confirmation.jsonl" \
  --site-id answer_query_v3 --cohort one_to_one --split confirmation \
  --cache-dir "$CACHE" --device-map auto --torch-dtype bfloat16 \
  --attention-backend sdpa

run_cpu answer_execution_plan \
  scripts/build_v5_answer_execution_plan.py \
  --capture-index "$EXT/representation/capture_primary/capture_index.jsonl" \
  --generations "$GEN" --model "$MODEL" \
  --output-dir "$EXT/answer_execution/plan" \
  --site-kind answer_query_v3 --site-id answer_query_v3 --rank 3

run_gpu answer_execution \
  scripts/run_realistic_niah_v5.py causal-patch \
  --model "$MODEL" --generations "$GEN" \
  --pairs "$EXT/answer_execution/plan/${MODEL}__answer_execution_pairs.jsonl" \
  --basis "$EXT/answer_execution/plan/${MODEL}__answer_query_v3_basis.npz" \
  --output "$EXT/answer_execution/trials_confirmation.jsonl" \
  --layer "$("$PY" -c "import json; print(json.load(open('$EXT/answer_execution/plan/${MODEL}__answer_execution_layer_selection.json'))['selected_layer'])")" \
  --receiver-site-id answer_query_v3 --donor-site-id answer_query_v3 \
  --max-new-tokens 16 --restartable \
  --cache-dir "$CACHE" --device-map auto --torch-dtype bfloat16 \
  --attention-backend sdpa

run_cpu analyze_answer_execution \
  scripts/analyze_v5_answer_execution.py \
  --trials "$EXT/answer_execution/trials_confirmation.jsonl" \
  --pairs "$EXT/answer_execution/plan/${MODEL}__answer_execution_pairs.jsonl" \
  --output-dir "$EXT/answer_execution/analysis"

for variant in pre_city_d1 pre_city_d2 pre_city_anchor; do
  mapfile -t PLAN_ROWS < <("$PY" - "$E4_PLAN" "$variant" <<'PY'
import sys
import pandas as pd

frame = pd.read_csv(sys.argv[1])
selected = frame.loc[frame["query_variant"].astype(str).eq(sys.argv[2])]
if selected.empty:
    raise SystemExit(f"no frozen E4 plan rows for {sys.argv[2]}")
for index in selected.index:
    print(int(index))
PY
)
  run_gpu "all_site_damage_${variant}" \
    scripts/run_realistic_niah_v5.py causal-pre-city-all-sites \
    --model "$MODEL" --generations "$GEN" --plan "$E4_PLAN" \
    --plan-rows "${PLAN_ROWS[@]}" --query-variant "$variant" \
    --cohort one_to_one \
    --worker-id "primary_gpu${GPU}" \
    --output "$GLOBAL/${variant}/trials_primary_confirmation.jsonl" \
    --cache-dir "$CACHE" --device-map auto --torch-dtype bfloat16 \
    --attention-backend sdpa

  run_gpu "all_site_damage_supplement_${variant}" \
    scripts/run_realistic_niah_v5.py causal-pre-city-all-sites \
    --model "$MODEL" --generations "$SUP_GEN" --plan "$E4_PLAN" \
    --plan-rows "${PLAN_ROWS[@]}" --query-variant "$variant" \
    --cohort one_to_one --allow-unregistered \
    --worker-id "primary_gpu${GPU}" \
    --output "$GLOBAL/${variant}/trials_supplement_n10_confirmation.jsonl" \
    --cache-dir "$CACHE" --device-map auto --torch-dtype bfloat16 \
    --attention-backend sdpa
done

run_cpu analyze_answer_query_head_damage \
  scripts/analyze_v5_greedy_head_damage.py \
  --trials \
  "$EXT/head_ablation/trials_primary_confirmation.jsonl" \
  "$EXT/head_ablation/trials_supplement_n10_confirmation.jsonl" \
  --output-dir "$EXT/head_ablation/analysis"

for variant in pre_city_d1 pre_city_d2 pre_city_anchor; do
  run_cpu "analyze_all_site_damage_${variant}" \
    scripts/analyze_v5_greedy_head_damage.py \
    --trials \
    "$GLOBAL/${variant}/trials_primary_confirmation.jsonl" \
    "$GLOBAL/${variant}/trials_supplement_n10_confirmation.jsonl" \
    --output-dir "$GLOBAL/${variant}/analysis"
done

printf 'complete\n' > "$STATUS"
echo "[$(date -Is)] COMPLETE $MODEL" | tee -a "$LOG/supervisor.log"

# The first model to finish releases its GPU into the other model's
# restartable all-site queue.  Claims make this safe whether or not the peer
# has reached the same stage yet; completed task shards are never recomputed.
if [[ "$MODEL" == "Qwen3-8B" ]]; then
  PEER=Gemma4-E4B
else
  PEER=Qwen3-8B
fi
PEER_STATUS=$RUN/$PEER/answer_query_extension_v3/logs/supervisor.status
PEER_LOG=$RUN/$PEER/answer_query_extension_v3/logs
if [[ ! -f "$PEER_STATUS" ]] || [[ "$(cat "$PEER_STATUS")" != "complete" ]]; then
  TAKEOVER_PID=$PEER_LOG/takeover_gpu${GPU}.pid
  if [[ ! -f "$TAKEOVER_PID" ]] || ! kill -0 "$(cat "$TAKEOVER_PID")" 2>/dev/null; then
    nohup bash scripts/run_v5_all_site_takeover_worker.sh "$PEER" "$GPU" \
      </dev/null >"$PEER_LOG/takeover_gpu${GPU}_stdout.log" 2>&1 &
    echo "[$(date -Is)] LAUNCHED peer takeover peer=$PEER gpu=$GPU pid=$!" \
      | tee -a "$LOG/supervisor.log"
  fi
fi
