#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 3 ]]; then
  echo "usage: $0 <index|bullet> <Qwen3-8B|Gemma4-E4B> <source|panel|plans|behavior|analyze|all>" >&2
  exit 2
fi

MODE=$1
MODEL=$2
PHASE=$3

case "$MODE" in
  index)
    PROMPT_MODE=enumeration_index
    ANCHOR_ROLE=post_marker
    ;;
  bullet)
    PROMPT_MODE=enumeration_bullet
    ANCHOR_ROLE=p0_item_end
    ;;
  *) echo "mode must be index or bullet" >&2; exit 2 ;;
esac

case "$MODEL" in
  Qwen3-8B)
    BANK_GRID=(32 64 80 96 112 128)
    SELECTION_METRIC=target_source_attention_mass
    REPORT_REFERENCE_K=128
    ;;
  Gemma4-E4B)
    BANK_GRID=(1 2 4 6 8)
    SELECTION_METRIC=source_attention_mass
    REPORT_REFERENCE_K=6
    ;;
  *) echo "unsupported model: $MODEL" >&2; exit 2 ;;
esac

case "$PHASE" in
  source|panel|plans|behavior|analyze|all) ;;
  *) echo "phase must be source, panel, plans, behavior, analyze, or all" >&2; exit 2 ;;
esac

ROOT=${V6_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}
PYTHON=${V6_PYTHON:-$ROOT/.venv/bin/python}
CACHE=${V6_CACHE:-$ROOT/.cache/huggingface}
RUN_ROOT=${V6_RUN_ROOT:-$ROOT/work/realistic_niah_v6/$PROMPT_MODE/$MODEL}
CONFIG=$ROOT/configs/realistic_niah_v6_${PROMPT_MODE}.json
REPORT_CONTRACT=$ROOT/configs/realistic_niah_v6_targeted_retrieval_report_contract.json
GENERATIONS=$RUN_ROOT/generation/generations.jsonl
COHORT_REGISTRY=$RUN_ROOT/replacement/discovery/selected_cells.jsonl
CAUSAL_ROOT=$RUN_ROOT/causal/targeted_retrieval/discovery_formal
SOURCE_ROOT=$CAUSAL_ROOT/source_writes/$ANCHOR_ROLE
PANEL_ROOT=$CAUSAL_ROOT/final_transition_panel
BEHAVIOR_REGISTRY=$PANEL_ROOT/behavior_anchor_registry.jsonl
LOG_ROOT=$CAUSAL_ROOT/logs

if [[ ! -x "$PYTHON" ]]; then
  echo "V6_PYTHON is not executable: $PYTHON" >&2
  exit 2
fi
if [[ ! -s "$REPORT_CONTRACT" ]]; then
  echo "missing targeted report contract: $REPORT_CONTRACT" >&2
  exit 2
fi
if [[ ! -s "$GENERATIONS" ]]; then
  echo "missing discovery generations: $GENERATIONS" >&2
  exit 2
fi
if [[ ! -s "$COHORT_REGISTRY" ]]; then
  echo "missing resolved discovery cohort: $COHORT_REGISTRY" >&2
  exit 2
fi

mkdir -p "$LOG_ROOT" "$CAUSAL_ROOT/locks"
cd "$ROOT"

exec 9>"$CAUSAL_ROOT/locks/supervisor.lock"
if ! flock -n 9; then
  echo "another $MODE/$MODEL targeted-retrieval supervisor owns the lock" >&2
  exit 75
fi

gpu_prefix=()
if [[ -n "${V6_CUDA_VISIBLE_DEVICES:-}" ]]; then
  gpu_prefix=(env CUDA_VISIBLE_DEVICES="$V6_CUDA_VISIBLE_DEVICES")
fi

run_logged() {
  local name=$1
  shift
  {
    echo "[$(date --iso-8601=seconds)] START $name"
    printf 'COMMAND'
    printf ' %q' "${gpu_prefix[@]}" "$@"
    printf '\n'
    "${gpu_prefix[@]}" "$@"
    echo "[$(date --iso-8601=seconds)] PASS $name"
  } 2>&1 | tee "$LOG_ROOT/$name.log"
}

run_source() {
  local resume_audit=$RUN_ROOT/quarantine/targeted_retrieval_completed_source_resume.recovery.json
  if [[ -s "$SOURCE_ROOT/manifest.json" && -s "$SOURCE_ROOT/v6_adapter_manifest.json" ]]; then
    run_logged validate_completed_source_write_resume \
      "$PYTHON" scripts/audit_realistic_niah_v6_completed_source_write_resume.py \
      --source "$SOURCE_ROOT" --model-label "$MODEL" \
      --prompt-mode "$PROMPT_MODE" --anchor-role "$ANCHOR_ROLE" \
      --output "$resume_audit"
    echo "[$(date --iso-8601=seconds)] SKIP completed source writes after fail-closed resume audit"
    return
  fi
  run_logged source_writes_${ANCHOR_ROLE} \
    "$PYTHON" scripts/run_realistic_niah_v6_causal.py \
    --v6-config "$CONFIG" --model-label "$MODEL" --phase discovery \
    --cohort-registry "$COHORT_REGISTRY" -- \
    causal-source-writes \
    --model "$MODEL" --cache-dir "$CACHE" \
    --device-map auto --torch-dtype bfloat16 --attention-backend sdpa \
    --generations "$GENERATIONS" --output "$SOURCE_ROOT" \
    --anchor-role "$ANCHOR_ROLE" --include-secondary
}

run_panel() {
  run_logged freeze_final_transition_panel \
    "$PYTHON" scripts/build_realistic_niah_v6_final_transition_panel.py \
    --v6-config "$CONFIG" --model "$MODEL" \
    --generations "$GENERATIONS" --cohort-registry "$COHORT_REGISTRY" \
    --source-writes "$SOURCE_ROOT" \
    --seed-role discovery --output "$PANEL_ROOT"
}

control_matching_for_k() {
  local k=$1
  if [[ "$MODEL" == Qwen3-8B && "$k" == 128 ]]; then
    printf 'global\n'
  else
    printf 'layer_matched\n'
  fi
}

random_condition_for_k() {
  local k=$1
  if [[ "$MODEL" == Qwen3-8B && "$k" == 128 ]]; then
    printf 'global_random\n'
  else
    printf 'layer_matched_random\n'
  fi
}

run_plans() {
  local k output matching resume_audit
  run_panel
  for k in "${BANK_GRID[@]}"; do
    output=$CAUSAL_ROOT/plans/k$k
    resume_audit=$output/v6_resume_audit.json
    if [[ -s "$output/retrieval_anchor_bank_plan.csv" && \
          -s "$output/causal_plan_audit.json" && \
          -s "$output/v6_adapter_manifest.json" ]]; then
      run_logged audit_plan_resume_k${k} \
        "$PYTHON" scripts/audit_realistic_niah_v6_targeted_plan_resume.py \
        --model "$MODEL" --bank-size "$k" --plan-dir "$output" \
        --report-contract "$REPORT_CONTRACT" --output "$resume_audit"
      echo "[$(date --iso-8601=seconds)] SKIP K=$k plan after fail-closed audit"
      continue
    fi
    matching=$(control_matching_for_k "$k")
    run_logged plan_k${k} \
      "$PYTHON" scripts/run_realistic_niah_v6_causal.py \
      --v6-config "$CONFIG" --model-label "$MODEL" --phase discovery \
      --cohort-registry "$COHORT_REGISTRY" -- \
      causal-plan \
      --source-writes "$SOURCE_ROOT" --output "$output" \
      --bank-size "$k" --anchor-role "$ANCHOR_ROLE" \
      --selection-metric "$SELECTION_METRIC" \
      --selection-eligibility-scope local \
      --selection-aggregation seed_event_mean \
      --random-control-matching "$matching" \
      --full-panel-plan
  done
}

run_behavior() {
  local k plan output random_condition
  if [[ ! -s "$BEHAVIOR_REGISTRY" ]]; then
    run_panel
  fi
  for k in "${BANK_GRID[@]}"; do
    plan=$CAUSAL_ROOT/plans/k$k/retrieval_anchor_bank_plan.csv
    output=$CAUSAL_ROOT/behavior/k$k
    random_condition=$(random_condition_for_k "$k")
    run_logged behavior_k${k} \
      "$PYTHON" scripts/run_realistic_niah_v6_causal.py \
      --v6-config "$CONFIG" --model-label "$MODEL" --phase discovery \
      --cohort-registry "$COHORT_REGISTRY" -- \
      causal-heads-behavior \
      --model "$MODEL" --cache-dir "$CACHE" \
      --device-map auto --torch-dtype bfloat16 --attention-backend sdpa \
      --generations "$GENERATIONS" --plan "$plan" --output "$output" \
      --anchor-role "$ANCHOR_ROLE" --include-secondary \
      --anchor-registry-input "$BEHAVIOR_REGISTRY" \
      --allow-selection-scope-bank-transfer \
      --evaluation-split discovery --counts 1 2 3 4 5 6 7 8 9 10 \
      --conditions clean selected_bank "$random_condition" \
      --limit 20 --anchor-sampling prompt_final_transition \
      --max-new-tokens 256 --decode-head-ablation-steps -1
  done
}

run_analysis() {
  run_logged analyze_dose_response \
    "$PYTHON" scripts/analyze_realistic_niah_v6_targeted_retrieval.py \
    --model "$MODEL" --prompt-mode "$PROMPT_MODE" \
    --causal-root "$CAUSAL_ROOT" --bank-sizes "${BANK_GRID[@]}" \
    --report-contract "$REPORT_CONTRACT" \
    --expected-seeds 20 --bootstrap-samples 10000 \
    --random-seed 20260828 --report-reference-k "$REPORT_REFERENCE_K" \
    --output "$CAUSAL_ROOT/analysis"
}

case "$PHASE" in
  source) run_source ;;
  panel) run_panel ;;
  plans) run_plans ;;
  behavior) run_behavior ;;
  analyze) run_analysis ;;
  all)
    run_source
    run_plans
    run_behavior
    run_analysis
    ;;
esac

printf 'PASS\n' >"$CAUSAL_ROOT/${PHASE}.COMPLETE"
