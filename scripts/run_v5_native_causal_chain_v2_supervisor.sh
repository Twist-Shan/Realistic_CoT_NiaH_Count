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
OLD_EXT=$MODEL_RUN/answer_query_extension_v3
ROOT=$MODEL_RUN/corrected_causal_chain_v2
MARKER=$ROOT/marker_adjacent_patch
NEXT=$ROOT/next_needle_ablation
ANSWER=$ROOT/answer_aggregation_factorial
EXECUTION=$ROOT/answer_execution_reanalysis
CACHE=$FS/cache/huggingface
LOG=$ROOT/logs
E4_PLAN=$MODEL_RUN/causal/pre_city_token/plan/causal_plan.csv

mkdir -p "$LOG" "$MARKER" "$NEXT" "$ANSWER" "$EXECUTION"
cd "$REPO"
STATUS=$LOG/supervisor.status
printf 'running\n' > "$STATUS"
printf '%s\n' "$$" > "$LOG/supervisor.pid"
trap 'code=$?; if [[ $code -ne 0 ]]; then printf "failed:%s\n" "$code" > "$STATUS"; fi' EXIT

if [[ "$MODEL" == "Qwen3-8B" ]]; then
  mapfile -t MARKER_LAYERS < <(seq 0 35)
elif [[ "$MODEL" == "Gemma4-E4B" ]]; then
  mapfile -t MARKER_LAYERS < <(seq 0 41)
else
  echo "unsupported model: $MODEL" >&2
  exit 2
fi

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

# Marker/pre-city patching mirrors the non-thinking adjacent-count
# counterfactual construction.  Discovery alone selects the residual layer;
# confirmation is then run at the frozen layer for each semantic query.
run_cpu marker_adjacent_plan \
  scripts/build_v5_marker_needle_patch_plan.py \
  --generations "$GEN" --model "$MODEL" --output-dir "$MARKER/plan"

PAIR_PLAN=$MARKER/plan/${MODEL}__marker_adjacent_pairs.jsonl
run_gpu marker_patch_discovery \
  scripts/run_realistic_niah_v5.py causal-marker-needle-patch \
  --model "$MODEL" --generations "$GEN" --pairs "$PAIR_PLAN" \
  --output "$MARKER/discovery/trials.jsonl" \
  --query-variants pre_city_d1 pre_city_d2 pre_city_anchor \
  --layers "${MARKER_LAYERS[@]}" --split discovery \
  --cache-dir "$CACHE" --device-map auto --torch-dtype bfloat16 \
  --attention-backend sdpa

run_cpu marker_patch_discovery_analysis \
  scripts/analyze_v5_marker_needle_patch.py \
  --trials "$MARKER/discovery/trials.jsonl" \
  --output-dir "$MARKER/discovery_analysis" --selection-only

SELECTION=$MARKER/discovery_analysis/discovery_frozen_layer_selection.csv
CONFIRMATION_TRIALS=()
for variant in pre_city_d1 pre_city_d2 pre_city_anchor; do
  layer=$("$PY" -c "import pandas as pd; f=pd.read_csv('$SELECTION'); q=f.loc[f['query_variant'].astype(str).eq('$variant')]; assert len(q)==1; print(int(q.iloc[0]['selected_layer']))")
  output=$MARKER/confirmation/${variant}__L${layer}.jsonl
  run_gpu "marker_patch_confirmation_${variant}" \
    scripts/run_realistic_niah_v5.py causal-marker-needle-patch \
    --model "$MODEL" --generations "$GEN" --pairs "$PAIR_PLAN" \
    --output "$output" --query-variants "$variant" --layers "$layer" \
    --split confirmation --cache-dir "$CACHE" --device-map auto \
    --torch-dtype bfloat16 --attention-backend sdpa
  CONFIRMATION_TRIALS+=("$output")
done

run_cpu marker_patch_final_analysis \
  scripts/analyze_v5_marker_needle_patch.py \
  --trials "$MARKER/discovery/trials.jsonl" "${CONFIRMATION_TRIALS[@]}" \
  --output-dir "$MARKER/analysis"

# Targeted-retrieval head ablation is evaluated locally in the trace.  It
# never scores final count: the endpoint is the actual greedy next-city token
# sequence, with both all-occurrence and clean-next-city-correct damage.
for variant in pre_city_d1 pre_city_d2 pre_city_anchor; do
  mapfile -t PLAN_ROWS < <("$PY" -c "import pandas as pd; f=pd.read_csv('$E4_PLAN'); q=f.loc[f['model_label'].astype(str).eq('$MODEL') & f['query_variant'].astype(str).eq('$variant')]; assert len(q)>0; [print(int(i)) for i in q.index]")
  run_gpu "next_needle_primary_${variant}" \
    scripts/run_realistic_niah_v5.py causal-pre-city-heads \
    --model "$MODEL" --generations "$GEN" --plan "$E4_PLAN" \
    --plan-rows "${PLAN_ROWS[@]}" --query-variant "$variant" \
    --cohort one_to_one_correct --split confirmation \
    --output "$NEXT/$variant/trials_primary_confirmation.jsonl" \
    --cache-dir "$CACHE" --device-map auto --torch-dtype bfloat16 \
    --attention-backend sdpa

  run_gpu "next_needle_supplement_${variant}" \
    scripts/run_realistic_niah_v5.py causal-pre-city-heads \
    --model "$MODEL" --generations "$SUP_GEN" --plan "$E4_PLAN" \
    --plan-rows "${PLAN_ROWS[@]}" --query-variant "$variant" \
    --cohort one_to_one_correct --split confirmation --allow-unregistered \
    --output "$NEXT/$variant/trials_supplement_n10_confirmation.jsonl" \
    --cache-dir "$CACHE" --device-map auto --torch-dtype bfloat16 \
    --attention-backend sdpa

  run_cpu "next_needle_analysis_${variant}" \
    scripts/analyze_v5_next_needle_ablation.py \
    --trials "$NEXT/$variant/trials_primary_confirmation.jsonl" \
    "$NEXT/$variant/trials_supplement_n10_confirmation.jsonl" \
    --output-dir "$NEXT/$variant/analysis"
done

# At answer_query_v3 the prompt-sequence bank, thinking-trace bank, and their
# joint union are reported as four behavioral conditions including clean.
# Existing prompt-only and trace-only greedy trials are immutable and reused;
# only the newly registered joint family is executed here.
run_cpu answer_factorial_plan \
  scripts/run_realistic_niah_v5.py answer-query-causal-plan \
  --attention "$OLD_EXT/head_ablation/attention_primary.csv" \
  --output "$ANSWER/plan"

ANSWER_PLAN=$ANSWER/plan/answer_query_causal_plan.csv
mapfile -t JOINT_ROWS < <("$PY" -c "import pandas as pd; f=pd.read_csv('$ANSWER_PLAN'); q=f.loc[f['model_label'].astype(str).eq('$MODEL') & f['mechanism'].astype(str).eq('answer_prompt_and_trace_aggregation')]; assert len(q)>0; [print(int(i)) for i in q.index]")
run_gpu answer_joint_primary \
  scripts/run_realistic_niah_v5.py causal-answer-query-heads \
  --model "$MODEL" --generations "$GEN" --plan "$ANSWER_PLAN" \
  --plan-rows "${JOINT_ROWS[@]}" --site-id answer_query_v3 \
  --cohort one_to_one_correct --split confirmation \
  --output "$ANSWER/trials_joint_primary_confirmation.jsonl" \
  --cache-dir "$CACHE" --device-map auto --torch-dtype bfloat16 \
  --attention-backend sdpa

run_gpu answer_joint_supplement \
  scripts/run_realistic_niah_v5.py causal-answer-query-heads \
  --model "$MODEL" --generations "$SUP_GEN" --plan "$ANSWER_PLAN" \
  --plan-rows "${JOINT_ROWS[@]}" --site-id answer_query_v3 \
  --cohort one_to_one_correct --split confirmation \
  --output "$ANSWER/trials_joint_supplement_n10_confirmation.jsonl" \
  --cache-dir "$CACHE" --device-map auto --torch-dtype bfloat16 \
  --attention-backend sdpa

run_cpu answer_factorial_analysis \
  scripts/analyze_v5_greedy_head_damage.py \
  --trials "$OLD_EXT/head_ablation/trials_primary_confirmation.jsonl" \
  "$OLD_EXT/head_ablation/trials_supplement_n10_confirmation.jsonl" \
  "$ANSWER/trials_joint_primary_confirmation.jsonl" \
  "$ANSWER/trials_joint_supplement_n10_confirmation.jsonl" \
  --output-dir "$ANSWER/analysis"

# Answer execution patching already used actual greedy count and correct-only
# receiver/donor pairs.  Reanalysis adds donor-vs-orthogonal specificity and
# Holm adjustment without mutating the registered trials.
run_cpu answer_execution_reanalysis \
  scripts/analyze_v5_answer_execution.py \
  --trials "$OLD_EXT/answer_execution/trials_confirmation.jsonl" \
  --pairs "$OLD_EXT/answer_execution/plan/${MODEL}__answer_execution_pairs.jsonl" \
  --output-dir "$EXECUTION/analysis"

printf 'complete\n' > "$STATUS"
echo "[$(date -Is)] COMPLETE $MODEL" | tee -a "$LOG/supervisor.log"
