#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 3 ]]; then
  echo "usage: $0 <index|bullet> <Qwen3-8B|Gemma4-E4B> <gpu-index>" >&2
  exit 2
fi
MODE=$1
MODEL=$2
GPU_INDEX=$3
case "$MODE" in index|bullet) ;; *) echo "unsupported mode: $MODE" >&2; exit 2 ;; esac
case "$MODEL" in
  Qwen3-8B)
    ANSWER_LAYERS=(0 5 10 15 20 25 30 35)
    SOURCE_LAYER=19
    RELAY_LAYER=26
    ;;
  Gemma4-E4B)
    ANSWER_LAYERS=(0 6 12 18 23 29 35 41)
    SOURCE_LAYER=16
    RELAY_LAYER=34
    ;;
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
REPLACEMENT_POOL=${V6_REPLACEMENT_POOL:-$RUN_BASE/replacement_seed_pool}
BASE_REPLACEMENT_POLICY=$ROOT/configs/realistic_niah_v6_replacement_policy.json
REPLACEMENT_POLICY=${V6_REPLACEMENT_POLICY:-$BASE_REPLACEMENT_POLICY}
COHORT_AMENDMENT=${V6_ANSWER_TRACE_COHORT_AMENDMENT:-}
PROMPT_MODE=enumeration_$MODE
MODEL_ROOT=$RUN_BASE/$PROMPT_MODE/$MODEL
CONFIG=$ROOT/configs/realistic_niah_v6_${PROMPT_MODE}.json
CONTRACT=$ROOT/configs/realistic_niah_v6_answer_trace_extension_v1.json
GENERATIONS=$MODEL_ROOT/generation/generations.jsonl
FREEZE=$MODEL_ROOT/freeze/confirmation_freeze.json
MECHANISM=$MODEL_ROOT/freeze/mechanism_frozen_confirmation.json
REPLACEMENT_STIMULI=$REPLACEMENT_POOL/stimuli.jsonl
BASE_COHORT_REGISTRY=$MODEL_ROOT/replacement/confirmation/selected_cells.jsonl
COHORT_ROOT=$MODEL_ROOT/replacement/confirmation_answer_trace
COHORT_REGISTRY=$COHORT_ROOT/selected_cells.jsonl
OUTPUT_ROOT=$MODEL_ROOT/causal/answer_trace_extension_v1
PLAN_ROOT=$OUTPUT_ROOT/answer_query_layer_sweep/plan
ANSWER_ROOT=$OUTPUT_ROOT/answer_query_layer_sweep
RELAY_ROOT=$OUTPUT_ROOT/terminal_relay_partial_confirmation
LOG_ROOT=$OUTPUT_ROOT/logs
LOCK_ROOT=$OUTPUT_ROOT/locks
COMPLETE=$OUTPUT_ROOT/extension.COMPLETE

export CUDA_VISIBLE_DEVICES=$GPU_INDEX
export HF_HOME=$CACHE
export TOKENIZERS_PARALLELISM=false
export HF_HUB_OFFLINE=${HF_HUB_OFFLINE:-1}
export TRANSFORMERS_OFFLINE=${TRANSFORMERS_OFFLINE:-1}
mkdir -p "$PLAN_ROOT" "$LOG_ROOT" "$LOCK_ROOT" "$CACHE"
exec > >(tee -a "$LOG_ROOT/supervisor.log") 2>&1
exec 9>"$LOCK_ROOT/supervisor.lock"
if ! flock -n 9; then
  echo "another $PROMPT_MODE/$MODEL answer-trace extension owns the lock" >&2
  exit 75
fi

for path in "$PYTHON" "$CONFIG" "$CONTRACT" "$GENERATIONS" "$FREEZE" \
  "$MECHANISM" "$REPLACEMENT_POLICY" "$REPLACEMENT_STIMULI" \
  "$BASE_COHORT_REGISTRY"; do
  test -s "$path" || { echo "missing extension input: $path" >&2; exit 4; }
done

if [[ "$REPLACEMENT_POLICY" != "$BASE_REPLACEMENT_POLICY" ]]; then
  test -n "$COHORT_AMENDMENT" || {
    echo "non-base answer/trace pool requires an explicit cohort amendment" >&2
    exit 4
  }
  test -s "$COHORT_AMENDMENT" || {
    echo "missing answer/trace cohort amendment: $COHORT_AMENDMENT" >&2
    exit 4
  }
  cd "$ROOT"
  "$PYTHON" - "$ROOT/src" "$COHORT_AMENDMENT" "$CONTRACT" "$REPLACEMENT_POLICY" \
    "$REPLACEMENT_POOL/manifest.json" "$REPLACEMENT_STIMULI" \
    "$PROMPT_MODE" "$MODEL" <<'PY'
import sys

sys.path.insert(0, sys.argv[1])

from realistic_niah_v6.answer_trace_extension import (
    validate_pool_exhaustion_amendment,
)

validate_pool_exhaustion_amendment(
    sys.argv[2],
    extension_contract_path=sys.argv[3],
    replacement_policy_path=sys.argv[4],
    pool_manifest_path=sys.argv[5],
    replacement_stimuli_path=sys.argv[6],
    prompt_mode=sys.argv[7],
    model_label=sys.argv[8],
)
print("PASS answer/trace pool-exhaustion amendment validation")
PY
fi

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
echo "[$(date --iso-8601=seconds)] START extension mode=$PROMPT_MODE model=$MODEL"

run_if_missing answer_trace_coherent_cohort "$COHORT_ROOT/manifest.json" \
  "$PYTHON" scripts/run_realistic_niah_v6_broad_panel_replacement.py \
    --v6-config "$CONFIG" --mechanism-config "$MECHANISM" \
    --replacement-policy "$REPLACEMENT_POLICY" \
    --panel-kind answer_trace \
    --answer-trace-extension-contract "$CONTRACT" \
    --replacement-stimuli "$REPLACEMENT_STIMULI" \
    --base-cohort-registry "$BASE_COHORT_REGISTRY" --model "$MODEL" \
    --phase answer_trace_confirmation \
    --generation-root "$MODEL_ROOT/generation" --output "$COHORT_ROOT" \
    --cache-dir "$CACHE" --device-map auto --torch-dtype bfloat16 \
    --attention-backend sdpa --confirmation-freeze "$FREEZE"
test -s "$COHORT_REGISTRY"

run_if_missing answer_query_plan "$PLAN_ROOT/plan_audit.json" \
  "$PYTHON" scripts/build_realistic_niah_v6_answer_query_layer_sweep_plan.py \
    --config "$CONFIG" --extension-contract "$CONTRACT" \
    --confirmation-freeze "$FREEZE" --generations "$GENERATIONS" \
    --cohort-registry "$COHORT_REGISTRY" --model "$MODEL" \
    --output-dir "$PLAN_ROOT"

run_if_missing answer_query_patch "$ANSWER_ROOT/trials.jsonl.v6_adapter.json" \
  "$PYTHON" scripts/run_realistic_niah_v6.py answer-query-patch \
    --config "$CONFIG" --extension-contract "$CONTRACT" \
    --confirmation-freeze "$FREEZE" --model "$MODEL" \
    --cache-dir "$CACHE" --device-map auto --torch-dtype bfloat16 \
    --attention-backend sdpa --generations "$GENERATIONS" \
    --seed-role confirmation --cohort-registry "$COHORT_REGISTRY" \
    --pairs "$PLAN_ROOT/pairs.jsonl" --output "$ANSWER_ROOT/trials.jsonl" \
    --layers "${ANSWER_LAYERS[@]}" \
    --conditions self_patch full_donor_patch \
    --receiver-site-id answer_query_v3 --donor-site-id answer_query_v3 \
    --max-new-tokens 16 --restartable

run_if_missing answer_query_analyze "$ANSWER_ROOT/analysis/v6_extension_audit.json" \
  "$PYTHON" scripts/analyze_realistic_niah_v6_answer_query_layer_sweep.py \
    --config "$CONFIG" --extension-contract "$CONTRACT" \
    --confirmation-freeze "$FREEZE" --cohort-registry "$COHORT_REGISTRY" \
    --model "$MODEL" --trials "$ANSWER_ROOT/trials.jsonl" \
    --pairs "$PLAN_ROOT/pairs.jsonl" --output-dir "$ANSWER_ROOT/analysis"

run_if_missing terminal_relay "$RELAY_ROOT/relay_confirmation/manifest.json" \
  "$PYTHON" scripts/run_realistic_niah_v6_count_stream.py \
    --v6-config "$CONFIG" --extension-contract "$CONTRACT" \
    --confirmation-freeze "$FREEZE" --cohort-registry "$COHORT_REGISTRY" \
    terminal-relay-mediation --mechanism-config "$MECHANISM" \
    --model "$MODEL" --cache-dir "$CACHE" --device-map auto \
    --torch-dtype bfloat16 --attention-backend sdpa \
    --generations "$GENERATIONS" --seed-role confirmation \
    --cohort one_to_one --row-panel trace_patch \
    --source-layer "$SOURCE_LAYER" --relay-layer "$RELAY_LAYER" \
    --geometry suffix8 --max-new-tokens 16 \
    --output "$RELAY_ROOT/relay_confirmation"

run_if_missing terminal_relay_analyze \
  "$RELAY_ROOT/relay_analysis_confirmation/v6_extension_audit.json" \
  "$PYTHON" scripts/analyze_realistic_niah_v6_terminal_relay_mediation.py \
    --config "$CONFIG" --extension-contract "$CONTRACT" \
    --confirmation-freeze "$FREEZE" --cohort-registry "$COHORT_REGISTRY" \
    --model "$MODEL" --trials "$RELAY_ROOT/relay_confirmation" \
    --output "$RELAY_ROOT/relay_analysis_confirmation" \
    --bootstrap-samples 10000 --random-seed 20260821

"$PYTHON" - "$PROMPT_MODE" "$MODEL" "$OUTPUT_ROOT" "$CONTRACT" \
  "$FREEZE" "$COHORT_REGISTRY" <<'PY'
import datetime as dt
import hashlib
import json
import pathlib
import sys

prompt_mode, model, output_raw, contract_raw, freeze_raw, cohort_raw = sys.argv[1:]
root, contract, freeze, cohort = map(
    pathlib.Path, (output_raw, contract_raw, freeze_raw, cohort_raw)
)
answer = json.loads(
    (root / "answer_query_layer_sweep/analysis/v6_extension_audit.json").read_text()
)
relay = json.loads(
    (root / "terminal_relay_partial_confirmation/relay_analysis_confirmation/v6_extension_audit.json").read_text()
)
value = {
    "schema_version": "realistic_niah_v6_answer_trace_extension_complete_v1",
    "status": "PASS_EXECUTION_COMPLETE",
    "prompt_mode": prompt_mode,
    "model_label": model,
    "answer_query_patching_status": answer["status"],
    "terminal_relay_execution_status": relay["status"],
    "terminal_relay_scientific_result": relay["scientific_result"],
    "partial_mediation_primary_pass": relay["partial_mediation_primary_pass"],
    "answer_query_only_secondary_pass": relay["answer_query_only_secondary_pass"],
    "complete_mediation_not_claimed": True,
    "seed_aliasing": False,
    "intervention_outcomes_used_for_selection": False,
    "extension_contract": str(contract.resolve()),
    "extension_contract_sha256": hashlib.sha256(contract.read_bytes()).hexdigest(),
    "confirmation_freeze_sha256": hashlib.sha256(freeze.read_bytes()).hexdigest(),
    "cohort_registry_sha256": hashlib.sha256(cohort.read_bytes()).hexdigest(),
    "completed_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
}
target = root / "extension_complete.json"
target.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
print(json.dumps(value, sort_keys=True))
PY

printf 'PASS\n' >"$COMPLETE"
echo "[$(date --iso-8601=seconds)] COMPLETE extension mode=$PROMPT_MODE model=$MODEL"
