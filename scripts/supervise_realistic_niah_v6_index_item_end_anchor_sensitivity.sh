#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 2 ]]; then
  echo "usage: $0 <Qwen3-8B|Gemma4-E4B> <gpu-index>" >&2
  exit 2
fi

MODEL=$1
GPU=$2

case "$MODEL" in
  Qwen3-8B)
    K=128
    SELECTION_METRIC=target_source_attention_mass
    CONTROL_MATCHING=global
    RANDOM_CONDITION=global_random
    ;;
  Gemma4-E4B)
    K=8
    SELECTION_METRIC=source_attention_mass
    CONTROL_MATCHING=layer_matched
    RANDOM_CONDITION=layer_matched_random
    ;;
  *) echo "unsupported model: $MODEL" >&2; exit 2 ;;
esac

ROOT=${V6_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}
PYTHON=${V6_PYTHON:-$ROOT/.venv/bin/python}
CACHE=${V6_CACHE:-$ROOT/.cache/huggingface}
V6_RUN_BASE=${V6_RUN_BASE:-$ROOT/work/realistic_niah_v6}
MODEL_ROOT=$V6_RUN_BASE/enumeration_index/$MODEL
CONFIG=$ROOT/configs/realistic_niah_v6_enumeration_index.json
CONTRACT=$ROOT/configs/realistic_niah_v6_index_item_end_anchor_sensitivity_v1.json
GENERATION_CONTAINER_AMENDMENT=$ROOT/configs/realistic_niah_v6_index_item_end_generation_container_amendment1.json
GENERATIONS=$MODEL_ROOT/generation/generations.jsonl
COHORT_REGISTRY=$MODEL_ROOT/replacement/discovery/selected_cells.jsonl
PRIMARY=$MODEL_ROOT/causal/targeted_retrieval/discovery_formal
SENSITIVITY=$MODEL_ROOT/causal/targeted_retrieval/anchor_sensitivity_item_end_exploratory_v1
P0_SOURCE=$SENSITIVITY/source_writes/p0_item_end
P0_PANEL=$SENSITIVITY/panels/p0_item_end
P0_PLAN_DIR=$SENSITIVITY/plans/p0_item_end/k$K
P0_PLAN=$P0_PLAN_DIR/retrieval_anchor_bank_plan.csv
P2_PLAN=$PRIMARY/plans/k$K/retrieval_anchor_bank_plan.csv
P0_REGISTRY=$P0_PANEL/behavior_anchor_registry.jsonl
P2_REGISTRY=$PRIMARY/final_transition_panel/behavior_anchor_registry.jsonl
LOG_ROOT=$SENSITIVITY/logs
COMPLETE=$SENSITIVITY/anchor_sensitivity.COMPLETE

for required in "$PYTHON" "$CONFIG" "$CONTRACT" "$GENERATIONS" \
  "$COHORT_REGISTRY" "$P2_PLAN" "$P2_REGISTRY"; do
  if [[ ! -s "$required" ]]; then
    echo "missing required sensitivity input: $required" >&2
    exit 2
  fi
done

mkdir -p "$LOG_ROOT" "$SENSITIVITY/locks" "$CACHE"
exec 9>"$SENSITIVITY/locks/supervisor.lock"
if ! flock -n 9; then
  echo "another $MODEL index item-end sensitivity supervisor owns the lock" >&2
  exit 75
fi

cd "$ROOT"
gpu_prefix=(env CUDA_VISIBLE_DEVICES="$GPU")
panel_extra=()
if [[ "$MODEL" == "Gemma4-E4B" ]]; then
  if [[ ! -s "$GENERATION_CONTAINER_AMENDMENT" ]]; then
    echo "missing Gemma generation-container amendment: $GENERATION_CONTAINER_AMENDMENT" >&2
    exit 2
  fi
  panel_extra=(--generation-container-amendment "$GENERATION_CONTAINER_AMENDMENT")
fi

run_logged() {
  local name=$1
  shift
  {
    echo "[$(date --iso-8601=seconds)] START $name"
    printf 'COMMAND'
    printf ' %q' "$@"
    printf '\n'
    "$@"
    echo "[$(date --iso-8601=seconds)] PASS $name"
  } 2>&1 | tee -a "$LOG_ROOT/$name.log"
}

if [[ ! -s "$P0_SOURCE/manifest.json" ]]; then
  run_logged source_writes_p0_item_end \
    "${gpu_prefix[@]}" "$PYTHON" scripts/run_realistic_niah_v6_causal.py \
    --v6-config "$CONFIG" --model-label "$MODEL" --phase diagnostic \
    --cohort-registry "$COHORT_REGISTRY" -- \
    causal-source-writes \
    --model "$MODEL" --cache-dir "$CACHE" \
    --device-map auto --torch-dtype bfloat16 --attention-backend sdpa \
    --generations "$GENERATIONS" --output "$P0_SOURCE" \
    --anchor-role p0_item_end --include-secondary
else
  run_logged audit_source_writes_p0_item_end_resume \
      "$PYTHON" scripts/audit_realistic_niah_v6_completed_source_write_resume.py \
      --source "$P0_SOURCE" --model-label "$MODEL" \
      --prompt-mode enumeration_index --anchor-role p0_item_end \
      --expected-phase diagnostic \
      --output "$SENSITIVITY/source_write_resume_audit.json"
fi

if [[ ! -s "$P0_PANEL/manifest.json" ]]; then
  run_logged build_p0_item_end_panel \
    "$PYTHON" scripts/build_realistic_niah_v6_index_item_end_sensitivity_panel.py \
    --v6-config "$CONFIG" --contract "$CONTRACT" --model "$MODEL" \
    --generations "$GENERATIONS" --cohort-registry "$COHORT_REGISTRY" \
    --source-writes "$P0_SOURCE" --output "$P0_PANEL" \
    "${panel_extra[@]}"
fi

if [[ ! -s "$P0_PLAN" ]]; then
  run_logged build_p0_item_end_k${K}_plan \
    "$PYTHON" scripts/run_realistic_niah_v6_causal.py \
    --v6-config "$CONFIG" --model-label "$MODEL" --phase diagnostic \
    --cohort-registry "$COHORT_REGISTRY" -- \
    causal-plan --source-writes "$P0_SOURCE" --output "$P0_PLAN_DIR" \
    --bank-size "$K" --anchor-role p0_item_end \
    --selection-metric "$SELECTION_METRIC" \
    --selection-eligibility-scope local \
    --selection-aggregation seed_event_mean \
    --random-control-matching "$CONTROL_MATCHING" \
    --full-panel-plan
fi

run_behavior() {
  local name=$1
  local plan=$2
  local role=$3
  local registry=$4
  shift 4
  local output=$SENSITIVITY/behavior/$name
  if [[ -s "$output/manifest.json" && -s "$output/v6_adapter_manifest.json" ]]; then
    echo "[$(date --iso-8601=seconds)] REUSE $name" | tee -a "$LOG_ROOT/$name.log"
    return
  fi
  run_logged "$name" \
    "${gpu_prefix[@]}" "$PYTHON" scripts/run_realistic_niah_v6_causal.py \
    --v6-config "$CONFIG" --model-label "$MODEL" --phase diagnostic \
    --cohort-registry "$COHORT_REGISTRY" -- \
    causal-heads-behavior \
    --model "$MODEL" --cache-dir "$CACHE" \
    --device-map auto --torch-dtype bfloat16 --attention-backend sdpa \
    --generations "$GENERATIONS" --plan "$plan" --output "$output" \
    --anchor-role "$role" --include-secondary \
    --anchor-registry-input "$registry" \
    --allow-selection-scope-bank-transfer \
    --evaluation-split discovery --counts 1 2 3 4 5 6 7 8 9 10 \
    --conditions "$@" --limit 20 \
    --anchor-sampling prompt_final_transition \
    --max-new-tokens 256 --decode-head-ablation-steps -1
}

# New behavior arms only.  The observed p2bank_at_p2 reference remains
# immutable under PRIMARY/behavior/kK and is read only by the final analysis.
run_behavior p0bank_at_p0 "$P0_PLAN" p0_item_end "$P0_REGISTRY" \
  clean selected_bank "$RANDOM_CONDITION"
run_behavior p2bank_at_p0 "$P2_PLAN" p0_item_end "$P0_REGISTRY" \
  selected_bank "$RANDOM_CONDITION"
run_behavior p0bank_at_p2 "$P0_PLAN" post_marker "$P2_REGISTRY" \
  selected_bank "$RANDOM_CONDITION"

run_logged analyze_anchor_sensitivity \
  "$PYTHON" scripts/analyze_realistic_niah_v6_index_item_end_anchor_sensitivity.py \
  --contract "$CONTRACT" --model "$MODEL" \
  --primary-root "$PRIMARY" --sensitivity-root "$SENSITIVITY" \
  --generations "$GENERATIONS" --cohort-registry "$COHORT_REGISTRY" \
  --output "$SENSITIVITY/analysis" \
  "${panel_extra[@]}"

printf 'PASS\n' >"$COMPLETE"
echo "[$(date --iso-8601=seconds)] ALL_COMPLETE $MODEL index item-end sensitivity"
