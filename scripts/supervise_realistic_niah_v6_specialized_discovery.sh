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
  index)
    PROMPT_MODE=enumeration_index
    TIMING=rank_before_city
    ANCHOR_ROLE=post_marker
    ;;
  bullet)
    PROMPT_MODE=enumeration_bullet
    TIMING=structural_item_end
    ANCHOR_ROLE=p0_item_end
    ;;
  *) echo "mode must be index or bullet" >&2; exit 2 ;;
esac
case "$MODEL" in
  Qwen3-8B) SOURCE_LAYER=19 ;;
  Gemma4-E4B) SOURCE_LAYER=16 ;;
  *) echo "unsupported model: $MODEL" >&2; exit 2 ;;
esac
if [[ ! "$GPU_INDEX" =~ ^[0-9]+$ ]]; then
  echo "gpu-index must be non-negative" >&2
  exit 2
fi

ROOT=${V6_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}
PYTHON=${V6_PYTHON:-$ROOT/.venv/bin/python}
CACHE=${V6_CACHE:-$ROOT/.cache/huggingface}
RUN_BASE=${V6_RUN_BASE:-$ROOT/work/realistic_niah_v6}
RUN_ROOT=$RUN_BASE/$PROMPT_MODE/$MODEL
CONFIG=$ROOT/configs/realistic_niah_v6_${PROMPT_MODE}.json
MECHANISM=$ROOT/configs/realistic_niah_v6_${PROMPT_MODE}_count_stream_dev.json
GENERATIONS=$RUN_ROOT/generation/generations.jsonl
COHORT_REGISTRY=$RUN_ROOT/replacement/discovery/selected_cells.jsonl
TARGET_ROOT=$RUN_ROOT/causal/targeted_retrieval/discovery_formal
SELECTION=$TARGET_ROOT/analysis/selection.json
PANEL_ROOT=$TARGET_ROOT/final_transition_panel
ANCHORS=$PANEL_ROOT/mode_panel.jsonl
TARGETED=$PANEL_ROOT/targeted_registry.jsonl
OUTPUT_ROOT=$RUN_ROOT/causal/specialized/discovery_formal
LOG_ROOT=$OUTPUT_ROOT/logs
SEED_ROLE_RECOVERY=$RUN_ROOT/quarantine/specialized_seed_role_spelling.recovery.json

for path in "$PYTHON" "$CONFIG" "$MECHANISM" "$GENERATIONS" \
  "$COHORT_REGISTRY" \
  "$SELECTION" "$ANCHORS" "$TARGETED"; do
  [[ -s "$path" ]] || { echo "missing specialized input: $path" >&2; exit 4; }
done

SELECTED_K=$("$PYTHON" -c \
  'import json,sys; print(int(json.load(open(sys.argv[1]))["selected_k"]))' \
  "$SELECTION")
SOURCE_BANK_PLAN=$TARGET_ROOT/plans/k$SELECTED_K/retrieval_anchor_bank_plan.csv
HEAD_UNIVERSE=$TARGET_ROOT/plans/k$SELECTED_K/crossfit_source_specific_head_ranking.csv
SPECIALIZED_PLAN_ROOT=$OUTPUT_ROOT/bank_plan
BANK_PLAN=$SPECIALIZED_PLAN_ROOT/retrieval_anchor_bank_plan.csv
SPECIALIZED_BANK_AUDIT=$SPECIALIZED_PLAN_ROOT/specialized_bank_plan_audit.json
for path in "$SOURCE_BANK_PLAN" "$HEAD_UNIVERSE"; do
  [[ -s "$path" ]] || { echo "missing selected-bank source: $path" >&2; exit 4; }
done

mkdir -p "$LOG_ROOT" "$OUTPUT_ROOT/locks"
cd "$ROOT"
exec 9>"$OUTPUT_ROOT/locks/supervisor.lock"
if ! flock -n 9; then
  echo "another $MODE/$MODEL specialized discovery supervisor owns the lock" >&2
  exit 75
fi

STRUCTURAL_SEED_ROLE_RESUME=0
PREEXISTING_TARGETED_WRITE_SHARDS=0
if [[ -s "$SEED_ROLE_RECOVERY" ]]; then
  STRUCTURAL_SEED_ROLE_RESUME=1
fi
if [[ -s "$OUTPUT_ROOT/stratified_ncc/v6_adapter_manifest.json" && \
      ! -s "$OUTPUT_ROOT/stratified_ncc/manifest.json" ]]; then
  STRUCTURAL_SEED_ROLE_RESUME=1
fi
if [[ ! -s "$OUTPUT_ROOT/discovery.COMPLETE" && \
      -s "$OUTPUT_ROOT/targeted_counter_write/manifest.json" ]] && \
    grep -Fq '"newly_completed": 0' \
      "$OUTPUT_ROOT/targeted_counter_write/manifest.json" && \
    grep -Fq '"resume_skipped": 20' \
      "$OUTPUT_ROOT/targeted_counter_write/manifest.json"; then
  STRUCTURAL_SEED_ROLE_RESUME=1
fi
if [[ -s "$LOG_ROOT/stratified-targeted-counter-ncc.log" ]] && \
    grep -Fq 'Stratified NCC development branch has only 0 seeds' \
      "$LOG_ROOT/stratified-targeted-counter-ncc.log"; then
  STRUCTURAL_SEED_ROLE_RESUME=1
fi
if [[ -d "$OUTPUT_ROOT/targeted_counter_write/shards" ]]; then
  PREEXISTING_TARGETED_WRITE_SHARDS=$(find \
    "$OUTPUT_ROOT/targeted_counter_write/shards" -maxdepth 1 \
    -type f -name '*.jsonl' | wc -l)
fi

run_logged() {
  local name=$1
  shift
  {
    echo "[$(date --iso-8601=seconds)] START $name"
    printf 'COMMAND env CUDA_VISIBLE_DEVICES=%q' "$GPU_INDEX"
    printf ' %q' "$@"
    printf '\n'
    env CUDA_VISIBLE_DEVICES="$GPU_INDEX" "$@"
    echo "[$(date --iso-8601=seconds)] PASS $name"
  } 2>&1 | tee "$LOG_ROOT/$name.log"
}

run_logged specialized-bank-plan \
  "$PYTHON" scripts/build_realistic_niah_v6_specialized_bank_plan.py \
  --selection "$SELECTION" --source-plan "$SOURCE_BANK_PLAN" \
  --head-universe "$HEAD_UNIVERSE" --model "$MODEL" \
  --prompt-mode "$PROMPT_MODE" --output "$SPECIALIZED_PLAN_ROOT"
for path in "$BANK_PLAN" "$SPECIALIZED_BANK_AUDIT" \
  "$SPECIALIZED_PLAN_ROOT/specialized_bank_plan.COMPLETE"; do
  [[ -s "$path" ]] || { echo "specialized bank-plan build failed: $path" >&2; exit 4; }
done

kernel() {
  local target=$1
  shift
  run_logged "$target" \
    "$PYTHON" scripts/run_realistic_niah_v6_kernel.py \
    --target "$target" --v6-config "$CONFIG" --phase discovery \
    --cohort-registry "$COHORT_REGISTRY" \
    --bank-selection "$SELECTION" -- "$@"
}

common_model_args=(
  --mechanism-config "$MECHANISM"
  --model "$MODEL"
  --cache-dir "$CACHE"
  --device-map auto
  --torch-dtype bfloat16
  --attention-backend sdpa
  --generations "$GENERATIONS"
  --seed-role development
)

kernel targeted-counter-write \
  "${common_model_args[@]}" \
  --anchor-registry "$ANCHORS" --targeted-registry "$TARGETED" \
  --bank-plan "$BANK_PLAN" --source-layer "$SOURCE_LAYER" --resume \
  --output "$OUTPUT_ROOT/targeted_counter_write"

kernel stratified-targeted-counter-ncc \
  "${common_model_args[@]}" \
  --timing "$TIMING" --panel "$ANCHORS" --bank-plan "$BANK_PLAN" --resume \
  --output "$OUTPUT_ROOT/stratified_ncc"

kernel targeted-counter-logit-margin \
  "${common_model_args[@]}" \
  --timing "$TIMING" --panel "$ANCHORS" --bank-plan "$BANK_PLAN" --resume \
  --output "$OUTPUT_ROOT/direct_count_logit_margin"

kernel targeted-counter-ncc \
  "${common_model_args[@]}" \
  --anchor-registry "$ANCHORS" --targeted-registry "$TARGETED" \
  --bank-plan "$BANK_PLAN" --source-layer "$SOURCE_LAYER" --resume \
  --output "$OUTPUT_ROOT/count_geometry_ncc"

run_logged terminal-token-state-bridge \
  "$PYTHON" scripts/run_realistic_niah_v6_kernel.py \
  --target terminal-token-state-bridge --v6-config "$CONFIG" \
  --phase discovery --cohort-registry "$COHORT_REGISTRY" -- \
  "${common_model_args[@]}" --anchor-registry "$ANCHORS" \
  --layer "$SOURCE_LAYER" --max-new-tokens 32 --resume \
  --output "$OUTPUT_ROOT/terminal_state_bridge"

kernel token-level-ablation \
  --mode answer --model "$MODEL" --cache-dir "$CACHE" \
  --device-map auto --torch-dtype bfloat16 --attention-backend sdpa \
  --generations "$GENERATIONS" --output "$OUTPUT_ROOT/token_ablation_answer" \
  --bank-plan "$BANK_PLAN" --bank-size "$SELECTED_K" \
  --anchor-registry "$TARGETED" --split discovery \
  --conditions clean prompt_all_blank prompt_records_blank trace_all_blank \
    prompt_and_trace_blank --run-greedy --max-new-tokens 32

kernel token-level-ablation \
  --mode targeting --model "$MODEL" --cache-dir "$CACHE" \
  --device-map auto --torch-dtype bfloat16 --attention-backend sdpa \
  --generations "$GENERATIONS" --output "$OUTPUT_ROOT/token_ablation_targeting" \
  --bank-plan "$BANK_PLAN" --bank-size "$SELECTED_K" \
  --anchor-role "$ANCHOR_ROLE" --anchor-registry "$TARGETED" \
  --split discovery --matched-control-repeats 3 \
  --conditions clean early_half_trace_blank cumulative_trace_blank \
    recent_transition_blank full_trace_blank \
    early_half_trace_matched_control cumulative_trace_matched_control \
    recent_transition_matched_control full_trace_matched_control \
  --run-greedy --max-new-tokens 32

"$PYTHON" - "$OUTPUT_ROOT" "$MODEL" "$PROMPT_MODE" "$TIMING" \
  "$SELECTION" "$BANK_PLAN" "$SPECIALIZED_BANK_AUDIT" \
  "$ANCHORS" "$TARGETED" "$STRUCTURAL_SEED_ROLE_RESUME" \
  "$PREEXISTING_TARGETED_WRITE_SHARDS" "$SEED_ROLE_RECOVERY" <<'PY'
import datetime as dt
import hashlib
import json
import os
import pathlib
import sys

root = pathlib.Path(sys.argv[1])
model, prompt_mode, timing = sys.argv[2:5]
selection, bank, bank_audit, anchors, targeted = map(pathlib.Path, sys.argv[5:10])
structural_seed_role_resume = bool(int(sys.argv[10]))
preexisting_targeted_write_shards = int(sys.argv[11])
seed_role_recovery_path = pathlib.Path(sys.argv[12])
required = {
    "targeted_counter_write": root / "targeted_counter_write/manifest.json",
    "stratified_ncc": root / "stratified_ncc/manifest.json",
    "direct_count_logit_margin": root / "direct_count_logit_margin/manifest.json",
    "count_geometry_ncc": root / "count_geometry_ncc/manifest.json",
    "terminal_state_bridge": root / "terminal_state_bridge/manifest.json",
    "token_ablation_answer": root / "token_ablation_answer/worker_00_manifest.json",
    "token_ablation_targeting": root / "token_ablation_targeting/worker_00_manifest.json",
}
missing = [str(path) for path in required.values() if not path.is_file()]
if missing:
    raise FileNotFoundError(missing)
if structural_seed_role_resume:
    targeted_write_manifest = json.loads(
        required["targeted_counter_write"].read_text()
    )
    if int(targeted_write_manifest.get("newly_completed", -1)) != 0:
        raise ValueError("seed-role recovery recomputed targeted-write shards")
    if int(targeted_write_manifest.get("resume_skipped", -1)) != (
        preexisting_targeted_write_shards
    ):
        raise ValueError("seed-role recovery did not reuse targeted-write shards")
    role_adapter_path = root / "stratified_ncc/v6_adapter_manifest.json"
    role_adapter = json.loads(role_adapter_path.read_text())
    role_audit = role_adapter.get("specialized_seed_role_adapter", {})
    if role_audit.get("status") != "APPLIED_V6_TO_LEGACY_ROLE_SPELLING":
        raise ValueError(f"unexpected specialized role adapter: {role_audit}")
    if role_audit.get("changed_field_only") != "stratified_ncc_seed_role":
        raise ValueError("specialized role adapter changed an unregistered field")
    if role_audit.get("seed_aliasing") is not False:
        raise ValueError("specialized role adapter aliases a source seed")
payload = {
    "schema_version": "realistic_niah_v6_specialized_discovery_complete_v2",
    "status": "DISCOVERY_COMPLETE",
    "model_label": model,
    "prompt_mode": prompt_mode,
    "mode_timing_stratum": timing,
    "completed_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
    "confirmation_opened": False,
    "intervention_outcomes_used_to_replace_rows": False,
    "structural_seed_role_spelling_recovery": structural_seed_role_resume,
    "selection": str(selection.resolve()),
    "selection_sha256": hashlib.sha256(selection.read_bytes()).hexdigest(),
    "bank_plan": str(bank.resolve()),
    "bank_plan_sha256": hashlib.sha256(bank.read_bytes()).hexdigest(),
    "specialized_bank_plan_adapter": {
        "path": str(bank_audit.resolve()),
        "sha256": hashlib.sha256(bank_audit.read_bytes()).hexdigest(),
    },
    "artifacts": {
        "specialized_bank_plan": {
            "path": str(bank.resolve()),
            "sha256": hashlib.sha256(bank.read_bytes()).hexdigest(),
        },
        "specialized_bank_plan_adapter": {
            "path": str(bank_audit.resolve()),
            "sha256": hashlib.sha256(bank_audit.read_bytes()).hexdigest(),
        },
    },
    "anchor_panel_sha256": hashlib.sha256(anchors.read_bytes()).hexdigest(),
    "targeted_registry_sha256": hashlib.sha256(targeted.read_bytes()).hexdigest(),
    "manifests": {
        name: {
            "path": str(path.resolve()),
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        }
        for name, path in required.items()
    },
}
path = root / "specialized_discovery_complete.json"
temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
temporary.replace(path)
if structural_seed_role_resume:
    role_adapter_path = root / "stratified_ncc/v6_adapter_manifest.json"
    role_plan_path = root / "stratified_ncc/frozen_row_plan.json"
    evidence = {
        "frozen_source_panel": anchors,
        "stratified_ncc_adapter": role_adapter_path,
        "stratified_ncc_row_plan": role_plan_path,
        "stratified_ncc_manifest": required["stratified_ncc"],
        "targeted_counter_write_manifest": required["targeted_counter_write"],
    }
    role_adapter = json.loads(role_adapter_path.read_text())
    role_audit = role_adapter["specialized_seed_role_adapter"]
    recovery = {
        "schema_version": "realistic_niah_v6_specialized_seed_role_recovery_v1",
        "status": "PASS_DISCOVERY_TO_DEVELOPMENT_ROLE_VIEW_RESUME",
        "prompt_mode": prompt_mode,
        "model_label": model,
        "reason": (
            "the frozen V6 timing panel labels its split discovery, while "
            "the inherited Native-thinking stratified NCC and direct-margin "
            "CLIs compare that field to their legacy spelling development"
        ),
        "repair": (
            "materialize a process-local panel view changing only "
            "stratified_ncc_seed_role from discovery to development"
        ),
        "source_panel_sha256": role_audit["source_panel_sha256"],
        "materialized_panel_sha256": role_audit["materialized_panel_sha256"],
        "changed_field_only": role_audit["changed_field_only"],
        "analysis_slot_count": int(role_audit["analysis_slot_count"]),
        "true_source_seed_count": int(role_audit["true_source_seed_count"]),
        "preexisting_targeted_write_shards": preexisting_targeted_write_shards,
        "targeted_write_resume_skipped": int(
            targeted_write_manifest["resume_skipped"]
        ),
        "completed_model_trials_recomputed": False,
        "sample_failure": False,
        "seed_replacement_triggered": False,
        "intervention_outcomes_used_for_repair": False,
        "seed_aliasing": False,
        "deletion_performed": False,
        "validated_files": {
            name: {
                "path": str(value.resolve()),
                "sha256": hashlib.sha256(value.read_bytes()).hexdigest(),
            }
            for name, value in evidence.items()
        },
    }
    seed_role_recovery_path.parent.mkdir(parents=True, exist_ok=True)
    recovery_tmp = seed_role_recovery_path.with_name(
        f".{seed_role_recovery_path.name}.{os.getpid()}.tmp"
    )
    recovery_tmp.write_text(
        json.dumps(recovery, indent=2, sort_keys=True) + "\n"
    )
    recovery_tmp.replace(seed_role_recovery_path)
print(json.dumps(payload, sort_keys=True))
PY

printf 'PASS\n' >"$OUTPUT_ROOT/discovery.COMPLETE"
