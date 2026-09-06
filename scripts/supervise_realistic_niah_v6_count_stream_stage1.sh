#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 3 ]]; then
  echo "usage: $0 <index|bullet> <Qwen3-8B|Gemma4-E4B> <gpu-index>" >&2
  exit 2
fi
MODE=$1
MODEL=$2
GPU_INDEX=$3

case "$MODE" in
  index|bullet) PROMPT_MODE=enumeration_$MODE ;;
  *) echo "mode must be index or bullet" >&2; exit 2 ;;
esac
case "$MODEL" in
  Qwen3-8B) SOURCE_LAYER=18; READOUT_LAYER=19 ;;
  Gemma4-E4B) SOURCE_LAYER=16; READOUT_LAYER=17 ;;
  *) echo "unsupported model: $MODEL" >&2; exit 2 ;;
esac
if [[ ! "$GPU_INDEX" =~ ^[0-9]+$ ]]; then
  echo "gpu-index must be a non-negative integer" >&2
  exit 2
fi

ROOT=${V6_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}
PYTHON=${V6_PYTHON:-$ROOT/.venv/bin/python}
CACHE=${V6_CACHE:-$ROOT/.cache/huggingface}
RUN_BASE=${V6_RUN_BASE:-$ROOT/work/realistic_niah_v6}
MODEL_ROOT=$RUN_BASE/$PROMPT_MODE/$MODEL
RUN_ROOT=$MODEL_ROOT/count_stream/discovery_formal
CONFIG=$ROOT/configs/realistic_niah_v6_${PROMPT_MODE}.json
MECHANISM=$ROOT/configs/realistic_niah_v6_${PROMPT_MODE}_count_stream_dev.json
RUNNER=$ROOT/scripts/run_realistic_niah_v6_count_stream.py
GENERATIONS=$MODEL_ROOT/generation/generations.jsonl
COHORT_REGISTRY=$MODEL_ROOT/replacement/discovery/selected_cells.jsonl
COHERENT_BROAD_REGISTRY=$MODEL_ROOT/replacement/discovery_broad_k/selected_cells.jsonl
CAPTURE_INDEX=$MODEL_ROOT/capture/formal/capture_index.jsonl
LOG=$RUN_ROOT/logs/stage1.log

mkdir -p "$RUN_ROOT/logs" "$RUN_ROOT/locks"
exec > >(tee -a "$LOG") 2>&1
exec 9>"$RUN_ROOT/locks/stage1.lock"
if ! flock -n 9; then
  echo "another $MODE/$MODEL V6 count-stream stage-1 owns the lock" >&2
  exit 75
fi
for path in "$PYTHON" "$CONFIG" "$MECHANISM" "$RUNNER" \
  "$GENERATIONS" "$COHORT_REGISTRY" "$CAPTURE_INDEX"; do
  test -s "$path" || { echo "missing input: $path" >&2; exit 4; }
done
test -s "$COHERENT_BROAD_REGISTRY" || {
  echo "missing coherent broad registry: $COHERENT_BROAD_REGISTRY" >&2
  exit 4
}
cd "$ROOT"

run_logged() {
  local label=$1
  shift
  echo "[$(date --iso-8601=seconds)] START $label"
  "$@"
  echo "[$(date --iso-8601=seconds)] PASS $label"
}

v6() {
  env CUDA_VISIBLE_DEVICES="$GPU_INDEX" HF_HUB_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1 "$PYTHON" "$RUNNER" \
    --v6-config "$CONFIG" --cohort-registry "$COHORT_REGISTRY" "$@"
}

v6_broad_k() {
  env CUDA_VISIBLE_DEVICES="$GPU_INDEX" HF_HUB_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1 "$PYTHON" "$RUNNER" \
    --v6-config "$CONFIG" --cohort-registry "$COHERENT_BROAD_REGISTRY" "$@"
}

TRACE_PLAN_DIR=$RUN_ROOT/trace_pair_plan
TRACE_PLAN=$TRACE_PLAN_DIR/trace_patch_pair_plan.csv
BASIS=$RUN_ROOT/running_basis.npz
TRACE_PATCH=$RUN_ROOT/trace_patch
BROAD_CAPTURE=$RUN_ROOT/broad_ranking_capture
TRACE_PLAN_BANK=$RUN_ROOT/broad_plan_trace
PROMPT_PLAN_BANK=$RUN_ROOT/broad_plan_prompt
TRACE_TRIALS=$RUN_ROOT/broad_k_grid_trace
PROMPT_TRIALS=$RUN_ROOT/broad_k_grid_prompt
FINAL_TRACE_SELECTION=$RUN_ROOT/k_selection_trace
FINAL_PROMPT_SELECTION=$RUN_ROOT/k_selection_prompt
REUSE_BROAD=0
if [[ -s "$FINAL_TRACE_SELECTION/k_selection_decision.json" && \
      -s "$FINAL_TRACE_SELECTION/manifest.json" && \
      -s "$FINAL_PROMPT_SELECTION/k_selection_decision.json" && \
      -s "$FINAL_PROMPT_SELECTION/manifest.json" ]]; then
  REUSE_BROAD=1
fi
STRUCTURAL_PROGRESS_RESUME=0
STRUCTURAL_OUTCOME_RESUME=0
PREEXISTING_TRACE_PATCH_SHARDS=0
if [[ -s "$TRACE_PATCH/v6_adapter_manifest.json" && \
      ! -s "$TRACE_PATCH/manifest.json" ]]; then
  STRUCTURAL_PROGRESS_RESUME=1
fi
if [[ -s "$LOG" ]] && grep -Fq "KeyError: 'from_occurrence'" "$LOG"; then
  STRUCTURAL_PROGRESS_RESUME=1
fi
if [[ -s "$LOG" ]] && \
    grep -Fq 'No finite registered trial outcomes remain' "$LOG"; then
  STRUCTURAL_OUTCOME_RESUME=1
fi
if [[ -d "$TRACE_PATCH/shards" ]]; then
  PREEXISTING_TRACE_PATCH_SHARDS=$(find "$TRACE_PATCH/shards" -maxdepth 1 \
    -type f -name '*.jsonl' | wc -l)
fi

run_logged plan_trace_patch v6 plan-trace-patch \
  --mechanism-config "$MECHANISM" --model "$MODEL" \
  --generations "$GENERATIONS" --cohort one_to_one \
  --output "$TRACE_PLAN_DIR"

if [[ ! -s "$BASIS" ]]; then
  run_logged fit_basis v6 fit-basis \
    --mechanism-config "$MECHANISM" --capture-index "$CAPTURE_INDEX" \
    --site-kind item_end --label occurrence --cohort one_to_one \
    --layers "$SOURCE_LAYER" "$READOUT_LAYER" --rank 3 \
    --random-seed 20260820 --output "$BASIS"
fi

if [[ "$REUSE_BROAD" -eq 0 ]]; then
  run_logged capture_broad v6 capture-broad \
    --mechanism-config "$MECHANISM" --model "$MODEL" --cache-dir "$CACHE" \
    --device-map auto --torch-dtype bfloat16 --attention-backend sdpa \
    --generations "$GENERATIONS" --seed-role development \
    --cohort parser_hit --row-panel broad_ranking --output "$BROAD_CAPTURE"

  for source in trace_items prompt_records; do
    if [[ "$source" == trace_items ]]; then
      plan_dir=$TRACE_PLAN_BANK
    else
      plan_dir=$PROMPT_PLAN_BANK
    fi
    run_logged plan_broad_$source v6 plan-broad \
      --mechanism-config "$MECHANISM" --captures "$BROAD_CAPTURE" \
      --model "$MODEL" --source-group "$source" --random-seed 20260820 \
      --output "$plan_dir"
  done
else
  echo "[$(date --iso-8601=seconds)] REUSE complete discovery broad-K decisions"
fi

run_k_grid() {
  local plan=$1
  local output=$2
  shift 2
  v6_broad_k broad-heads --mechanism-config "$MECHANISM" --model "$MODEL" \
    --cache-dir "$CACHE" --device-map auto --torch-dtype bfloat16 \
    --attention-backend sdpa --generations "$GENERATIONS" \
    --seed-role development --cohort parser_hit \
    --row-panel broad_k_selection --plan "$plan" --bank-sizes "$@" \
    --skip-greedy --output "$output"
}

if [[ "$REUSE_BROAD" -eq 0 ]]; then
  run_logged broad_grid_trace run_k_grid \
    "$TRACE_PLAN_BANK/answer_broad_head_plan.csv" "$TRACE_TRIALS" \
    1 2 4 8 16 32
  run_logged broad_grid_prompt run_k_grid \
    "$PROMPT_PLAN_BANK/answer_broad_head_plan.csv" "$PROMPT_TRIALS" \
    1 2 4 8 16 32
fi

select_k() {
  local source=$1
  local plan=$2
  local trials=$3
  local output=$4
  shift 4
  v6_broad_k select-broad-k --mechanism-config "$MECHANISM" --model "$MODEL" \
    --source-group "$source" --plan "$plan" --trials "$trials" "$@" \
    --random-seed 20260820 --output "$output"
}

decision_status() {
  "$PYTHON" -c \
    'import json,sys; print(json.load(open(sys.argv[1]))["status"])' "$1"
}

if [[ "$REUSE_BROAD" -eq 0 ]]; then
  run_logged select_trace select_k trace_items \
    "$TRACE_PLAN_BANK/answer_broad_head_plan.csv" "$TRACE_TRIALS" \
    "$FINAL_TRACE_SELECTION"
  run_logged select_prompt select_k prompt_records \
    "$PROMPT_PLAN_BANK/answer_broad_head_plan.csv" "$PROMPT_TRIALS" \
    "$FINAL_PROMPT_SELECTION"
fi

extend_if_needed() {
  local source=$1
  local base_trials=$2
  local base_selection=$3
  local result_variable=$4
  if [[ "$(decision_status "$base_selection/k_selection_decision.json")" \
        != requires_boundary_extension ]]; then
    printf -v "$result_variable" '%s' "$base_selection"
    return
  fi
  local extended_plan=$RUN_ROOT/broad_plan_${source}_with_k64
  local trials_k64=$RUN_ROOT/broad_k64_${source}
  local extended_selection=$RUN_ROOT/k_selection_${source}_with_k64
  run_logged plan_${source}_k64 v6 plan-broad \
    --mechanism-config "$MECHANISM" --captures "$BROAD_CAPTURE" \
    --model "$MODEL" --source-group "$source" \
    --bank-sizes 1 2 4 8 16 32 64 --random-seed 20260820 \
    --output "$extended_plan"
  run_logged grid_${source}_k64 run_k_grid \
    "$extended_plan/answer_broad_head_plan.csv" "$trials_k64" 64
  run_logged select_${source}_k64 select_k "$source" \
    "$extended_plan/answer_broad_head_plan.csv" "$base_trials" \
    "$extended_selection" "$trials_k64"
  printf -v "$result_variable" '%s' "$extended_selection"
}

reuse_or_extend() {
  local source=$1
  local base_trials=$2
  local base_selection=$3
  local result_variable=$4
  if [[ "$REUSE_BROAD" -eq 0 ]]; then
    extend_if_needed "$source" "$base_trials" "$base_selection" "$result_variable"
    return
  fi
  if [[ "$(decision_status "$base_selection/k_selection_decision.json")" \
        == requires_boundary_extension ]]; then
    local extended_selection=$RUN_ROOT/k_selection_${source}_with_k64
    test -s "$extended_selection/k_selection_decision.json"
    test -s "$extended_selection/manifest.json"
    printf -v "$result_variable" '%s' "$extended_selection"
  else
    printf -v "$result_variable" '%s' "$base_selection"
  fi
}

reuse_or_extend trace_items "$TRACE_TRIALS" \
  "$FINAL_TRACE_SELECTION" FINAL_TRACE_SELECTION
reuse_or_extend prompt_records "$PROMPT_TRIALS" \
  "$FINAL_PROMPT_SELECTION" FINAL_PROMPT_SELECTION

run_logged trace_patch v6 trace-patch \
  --mechanism-config "$MECHANISM" --model "$MODEL" --cache-dir "$CACHE" \
  --device-map auto --torch-dtype bfloat16 --attention-backend sdpa \
  --generations "$GENERATIONS" --seed-role development \
  --cohort one_to_one --row-panel trace_patch --pair-plan "$TRACE_PLAN" \
  --basis "$BASIS" --layer "$SOURCE_LAYER" \
  --readout-layers "$READOUT_LAYER" \
  --conditions clean self_patch full_donor_patch progress_projected_patch \
    norm_matched_orthogonal_patch --random-seed 20260820 --skip-greedy \
  --output "$TRACE_PATCH"

run_logged analyze_trace_patch v6 analyze \
  --mechanism-config "$MECHANISM" --trials "$TRACE_PATCH" \
  --experiment-ids trace_intermediate_state_patching \
    trace_terminal_state_patching \
  --outcome donor_vs_receiver_city_log_odds --strata donor_direction \
  --random-seed 20260820 --output "$RUN_ROOT/trace_patch_analysis"

"$PYTHON" - "$RUN_ROOT" "$MODEL" "$PROMPT_MODE" \
  "$FINAL_TRACE_SELECTION" "$FINAL_PROMPT_SELECTION" "$REUSE_BROAD" \
  "$STRUCTURAL_PROGRESS_RESUME" "$STRUCTURAL_OUTCOME_RESUME" \
  "$PREEXISTING_TRACE_PATCH_SHARDS" <<'PY'
import datetime as dt
import hashlib
import json
import os
import pathlib
import sys

root = pathlib.Path(sys.argv[1])
model, prompt_mode = sys.argv[2:4]
trace_selection, prompt_selection = map(pathlib.Path, sys.argv[4:6])
reused_broad = bool(int(sys.argv[6]))
structural_resume = bool(int(sys.argv[7]))
outcome_resume = bool(int(sys.argv[8]))
preexisting_trace_patch_shards = int(sys.argv[9])
pair_manifest = json.loads((root / "trace_pair_plan/manifest.json").read_text())
trace_manifest = json.loads((root / "trace_patch/manifest.json").read_text())
analysis_manifest = json.loads(
    (root / "trace_patch_analysis/manifest.json").read_text()
)
plan_rows = (root / "trace_pair_plan/trace_patch_pair_plan.csv").read_text()
pair_count = int(pair_manifest["pair_count"])
if int(trace_manifest["completed_shards"]) != pair_count:
    raise ValueError("trace-patch shard count does not match the frozen plan")
if outcome_resume:
    if int(trace_manifest.get("newly_completed", -1)) != 0:
        raise ValueError("terminal-outcome resume recomputed model trials")
    if int(trace_manifest.get("resume_skipped", -1)) != pair_count:
        raise ValueError("terminal-outcome resume did not reuse every model trial")
    effective = analysis_manifest.get("effective_outcomes_by_experiment", {})
    if effective != {
        "trace_intermediate_state_patching": "donor_vs_receiver_city_log_odds",
        "trace_terminal_state_patching": "correct_count_margin",
    }:
        raise ValueError(f"unexpected trace outcome routing: {effective}")
decisions = {
    "trace_items": json.loads(
        (trace_selection / "k_selection_decision.json").read_text()
    ),
    "prompt_records": json.loads(
        (prompt_selection / "k_selection_decision.json").read_text()
    ),
}
allowed = {"frozen_for_confirmation", "no_positive_discovery_bank"}
if any(value["status"] not in allowed for value in decisions.values()):
    raise ValueError(f"unfrozen broad-head decision: {decisions}")
payload = {
    "schema_version": "realistic_niah_v6_count_stream_stage1_v1",
    "status": "DISCOVERY_COMPLETE",
    "model_label": model,
    "prompt_mode": prompt_mode,
    "completed_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
    "confirmation_opened": False,
    "complete_discovery_broad_k_decisions_reused": reused_broad,
    "broad_k_reselection_performed_on_resume": False if reused_broad else None,
    "structural_progress_dispatch_recovery": structural_resume,
    "structural_terminal_outcome_recovery": outcome_resume,
    "pair_count": pair_count,
    "cell_count": int(pair_manifest["cell_count"]),
    "trace_plan_sha256": hashlib.sha256(plan_rows.encode()).hexdigest(),
    "trace_patch_completed_shards": int(trace_manifest["completed_shards"]),
    "source_layer": int(trace_manifest["patch_layer"]),
    "readout_layers": trace_manifest["readout_layers"],
    "broad_head_decisions": decisions,
    "trace_selection_dir": str(trace_selection.resolve()),
    "prompt_selection_dir": str(prompt_selection.resolve()),
}
path = root / "stage1_complete.json"
tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
tmp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
tmp.replace(path)
if structural_resume:
    model_root = root.parents[1]
    evidence = {
        "trace_patch_manifest": root / "trace_patch/manifest.json",
        "trace_patch_adapter": root / "trace_patch/v6_adapter_manifest.json",
        "trace_patch_analysis": root / "trace_patch_analysis/manifest.json",
    }
    recovery = {
        "schema_version": "realistic_niah_v6_count_stream_structural_resume_v1",
        "status": "PASS_REGISTERED_PROGRESS_TRANSITION_RESUME",
        "prompt_mode": prompt_mode,
        "model_label": model,
        "reason": (
            "the frozen late-V5 trace-patch kernel requires registry-shaped "
            "progress-transition fields, while its public dispatcher routed "
            "the first attempt through the legacy marker-only compiler"
        ),
        "repair": (
            "process-local routing to the existing frozen grammar-aware "
            "progress compiler; numerical patching and estimands unchanged"
        ),
        "preexisting_completed_trace_patch_shards_at_final_resume": preexisting_trace_patch_shards,
        "first_failed_attempt_completed_progress_lines": (
            (root / "logs/stage1.log").read_text().split(
                "KeyError: 'from_occurrence'", 1
            )[0].count("[count-stream trace-patch]")
        ),
        "complete_discovery_broad_k_decisions_reused": reused_broad,
        "completed_model_trials_recomputed": False,
        "sample_failure": False,
        "seed_replacement_triggered": False,
        "intervention_outcomes_used_for_repair": False,
        "deletion_performed": False,
        "validated_files": {
            name: {
                "path": str(value.resolve()),
                "sha256": hashlib.sha256(value.read_bytes()).hexdigest(),
            }
            for name, value in evidence.items()
        },
    }
    recovery_path = model_root / (
        "quarantine/count_stream_progress_transition.recovery.json"
    )
    recovery_path.parent.mkdir(parents=True, exist_ok=True)
    recovery_tmp = recovery_path.with_name(
        f".{recovery_path.name}.{os.getpid()}.tmp"
    )
    recovery_tmp.write_text(
        json.dumps(recovery, indent=2, sort_keys=True) + "\n"
    )
    recovery_tmp.replace(recovery_path)
if outcome_resume:
    model_root = root.parents[1]
    evidence = {
        "trace_patch_manifest": root / "trace_patch/manifest.json",
        "trace_patch_adapter": root / "trace_patch/v6_adapter_manifest.json",
        "trace_patch_analysis": root / "trace_patch_analysis/manifest.json",
        "trace_estimands": root / "trace_patch_analysis/estimands.csv",
    }
    recovery = {
        "schema_version": "realistic_niah_v6_count_stream_terminal_outcome_recovery_v1",
        "status": "PASS_REGISTERED_TERMINAL_OUTCOME_ROUTING_RESUME",
        "prompt_mode": prompt_mode,
        "model_label": model,
        "reason": (
            "the inherited single-outcome analyzer applied the registered "
            "local next-city endpoint to the answer-only terminal panel"
        ),
        "repair": (
            "process-local routing by frozen experiment topology: local pairs "
            "retain donor_vs_receiver_city_log_odds and terminal pairs use "
            "correct_count_margin before unchanged V5 contrast aggregation"
        ),
        "effective_outcomes_by_experiment": (
            analysis_manifest["effective_outcomes_by_experiment"]
        ),
        "preexisting_completed_trace_patch_shards": preexisting_trace_patch_shards,
        "trace_patch_resume_skipped": int(trace_manifest["resume_skipped"]),
        "completed_model_trials_recomputed": False,
        "analysis_only_recomputed": True,
        "sample_failure": False,
        "seed_replacement_triggered": False,
        "intervention_effect_magnitudes_used_for_routing": False,
        "deletion_performed": False,
        "validated_files": {
            name: {
                "path": str(value.resolve()),
                "sha256": hashlib.sha256(value.read_bytes()).hexdigest(),
            }
            for name, value in evidence.items()
        },
    }
    recovery_path = model_root / (
        "quarantine/count_stream_terminal_outcome_routing.recovery.json"
    )
    recovery_path.parent.mkdir(parents=True, exist_ok=True)
    recovery_tmp = recovery_path.with_name(
        f".{recovery_path.name}.{os.getpid()}.tmp"
    )
    recovery_tmp.write_text(
        json.dumps(recovery, indent=2, sort_keys=True) + "\n"
    )
    recovery_tmp.replace(recovery_path)
print(json.dumps(payload, sort_keys=True))
PY

printf 'PASS\n' >"$RUN_ROOT/stage1.COMPLETE"
