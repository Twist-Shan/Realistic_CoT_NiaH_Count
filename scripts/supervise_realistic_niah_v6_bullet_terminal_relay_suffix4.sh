#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 2 ]]; then
  echo "usage: $0 <Qwen3-8B|Gemma4-E4B> <gpu-index>" >&2
  exit 2
fi
MODEL=$1
GPU_INDEX=$2
case "$MODEL" in
  Qwen3-8B) SOURCE_LAYER=19; RELAY_LAYER=26 ;;
  Gemma4-E4B) SOURCE_LAYER=16; RELAY_LAYER=34 ;;
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
PROMPT_MODE=enumeration_bullet
MODEL_ROOT=$RUN_BASE/$PROMPT_MODE/$MODEL
CONFIG=$ROOT/configs/realistic_niah_v6_enumeration_bullet.json
CONTRACT=$ROOT/configs/realistic_niah_v6_answer_trace_extension_v1.json
AMENDMENT=$ROOT/configs/realistic_niah_v6_bullet_terminal_relay_suffix4_amendment_v1.json
GENERATIONS=$MODEL_ROOT/generation/generations.jsonl
FREEZE=$MODEL_ROOT/freeze/confirmation_freeze.json
MECHANISM=$MODEL_ROOT/freeze/mechanism_frozen_confirmation.json
COHORT_REGISTRY=$MODEL_ROOT/replacement/confirmation_answer_trace/selected_cells.jsonl
OUTPUT_ROOT=$MODEL_ROOT/causal/answer_trace_extension_v1
RELAY_ROOT=$OUTPUT_ROOT/terminal_relay_partial_confirmation_suffix4
TRIAL_ROOT=$RELAY_ROOT/relay_confirmation
ANALYSIS_ROOT=$RELAY_ROOT/relay_analysis_confirmation
LOG_ROOT=$OUTPUT_ROOT/logs
LOCK_ROOT=$OUTPUT_ROOT/locks
COMPLETE=$RELAY_ROOT/suffix4_replication_complete.json

export CUDA_VISIBLE_DEVICES=$GPU_INDEX
export HF_HOME=$CACHE
export TOKENIZERS_PARALLELISM=false
export HF_HUB_OFFLINE=${HF_HUB_OFFLINE:-1}
export TRANSFORMERS_OFFLINE=${TRANSFORMERS_OFFLINE:-1}
mkdir -p "$LOG_ROOT" "$LOCK_ROOT" "$CACHE"
exec > >(tee -a "$LOG_ROOT/terminal_relay_suffix4.log") 2>&1
exec 9>"$LOCK_ROOT/terminal_relay_suffix4.lock"
if ! flock -n 9; then
  echo "another Bullet/$MODEL suffix4 relay replication owns the lock" >&2
  exit 75
fi

for path in "$PYTHON" "$CONFIG" "$CONTRACT" "$AMENDMENT" "$GENERATIONS" \
  "$FREEZE" "$MECHANISM" "$COHORT_REGISTRY"; do
  test -s "$path" || { echo "missing suffix4 replication input: $path" >&2; exit 4; }
done

run_if_missing() {
  local label=$1
  local artifact=$2
  shift 2
  if [[ -s "$artifact" ]]; then
    echo "[$(date --iso-8601=seconds)] REUSE $label artifact=$artifact"
    return 0
  fi
  echo "[$(date --iso-8601=seconds)] START $label gpu=$CUDA_VISIBLE_DEVICES"
  "$@"
  test -s "$artifact"
  echo "[$(date --iso-8601=seconds)] PASS $label"
}

cd "$ROOT"
echo "[$(date --iso-8601=seconds)] START Bullet/$MODEL task-adapted suffix4 relay"

run_if_missing terminal_relay_suffix4 "$TRIAL_ROOT/manifest.json" \
  "$PYTHON" scripts/run_realistic_niah_v6_count_stream.py \
    --v6-config "$CONFIG" --extension-contract "$CONTRACT" \
    --relay-geometry-amendment "$AMENDMENT" \
    --confirmation-freeze "$FREEZE" --cohort-registry "$COHORT_REGISTRY" \
    terminal-relay-mediation --mechanism-config "$MECHANISM" \
    --model "$MODEL" --cache-dir "$CACHE" --device-map auto \
    --torch-dtype bfloat16 --attention-backend sdpa \
    --generations "$GENERATIONS" --seed-role confirmation \
    --cohort one_to_one --row-panel trace_patch \
    --source-layer "$SOURCE_LAYER" --relay-layer "$RELAY_LAYER" \
    --geometry suffix4 --max-new-tokens 16 \
    --output "$TRIAL_ROOT"

run_if_missing terminal_relay_suffix4_analyze \
  "$ANALYSIS_ROOT/v6_extension_audit.json" \
  "$PYTHON" scripts/analyze_realistic_niah_v6_terminal_relay_mediation.py \
    --config "$CONFIG" --extension-contract "$CONTRACT" \
    --relay-geometry-amendment "$AMENDMENT" \
    --confirmation-freeze "$FREEZE" --cohort-registry "$COHORT_REGISTRY" \
    --model "$MODEL" --trials "$TRIAL_ROOT" --output "$ANALYSIS_ROOT" \
    --bootstrap-samples 10000 --random-seed 20260821

"$PYTHON" - "$MODEL" "$CONTRACT" "$AMENDMENT" "$TRIAL_ROOT" \
  "$ANALYSIS_ROOT" "$COMPLETE" <<'PY'
import datetime as dt
import hashlib
import json
import pathlib
import sys

model, contract_raw, amendment_raw, trials_raw, analysis_raw, complete_raw = sys.argv[1:]
contract, amendment, trials, analysis, complete = map(
    pathlib.Path,
    (contract_raw, amendment_raw, trials_raw, analysis_raw, complete_raw),
)
trial_manifest = json.loads((trials / "manifest.json").read_text())
audit = json.loads((analysis / "v6_extension_audit.json").read_text())
if trial_manifest["patch_geometry"] != "suffix4" or audit["relay_geometry"] != "suffix4":
    raise SystemExit("suffix4 geometry audit failed")
if audit["relay_scientific_label"] != "post_hoc_task_adapted_bullet_relay_replication":
    raise SystemExit("suffix4 evidence label audit failed")
value = {
    "schema_version": "realistic_niah_v6_bullet_terminal_relay_suffix4_complete_v1",
    "status": "PASS_EXECUTION_COMPLETE",
    "prompt_mode": "enumeration_bullet",
    "model_label": model,
    "relay_geometry": "suffix4",
    "scientific_label": "post_hoc_task_adapted_bullet_relay_replication",
    "relay_estimable": audit["relay_estimable"],
    "scientific_result": audit["scientific_result"],
    "partial_mediation_primary_pass": audit["partial_mediation_primary_pass"],
    "original_suffix8_artifacts_preserved": True,
    "extension_contract_sha256": hashlib.sha256(contract.read_bytes()).hexdigest(),
    "relay_geometry_amendment_sha256": hashlib.sha256(amendment.read_bytes()).hexdigest(),
    "trial_manifest_sha256": hashlib.sha256((trials / "manifest.json").read_bytes()).hexdigest(),
    "analysis_audit_sha256": hashlib.sha256((analysis / "v6_extension_audit.json").read_bytes()).hexdigest(),
    "completed_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
}
complete.parent.mkdir(parents=True, exist_ok=True)
temporary = complete.with_name(f".{complete.name}.tmp")
temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
temporary.replace(complete)
print(json.dumps(value, sort_keys=True))
PY

echo "[$(date --iso-8601=seconds)] COMPLETE Bullet/$MODEL suffix4 relay"
