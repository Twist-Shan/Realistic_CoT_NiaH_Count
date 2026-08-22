#!/usr/bin/env bash
set -euo pipefail

MODEL="${1:?usage: $0 MODEL}"
if [[ "$MODEL" != "Qwen3-8B" && "$MODEL" != "Gemma4-E4B" ]]; then
  echo "unsupported model: $MODEL" >&2
  exit 2
fi

CODE_ROOT="${CODE_ROOT:-/home/ubuntu/Realistic_CoT_NiaH_Count}"
PYTHON="${PYTHON:-$CODE_ROOT/.venv/bin/python}"
RUN_ROOT="${RUN_ROOT:-$CODE_ROOT/work/v5_native_count_stream/integrated_mediator_restoration_20d10c_20260821/$MODEL}"
RUNNER="$CODE_ROOT/scripts/run_realistic_niah_v5_count_stream.py"
ANALYZER="$CODE_ROOT/scripts/analyze_realistic_niah_v5_integrated_mediator_restoration.py"
DEV_MECHANISM="$CODE_ROOT/configs/realistic_niah_v5_native_count_stream_dev.json"
CONFIRM_MECHANISM="$CODE_ROOT/configs/realistic_niah_v5_native_count_stream_confirmation_v1.json"
V5_CONFIG="$CODE_ROOT/configs/realistic_niah_v5.json"
GENERATIONS="$CODE_ROOT/work/v5_trace_parser_v2/${MODEL}_generations_reparsed.jsonl"
TARGETED_ROOT="${TARGETED_ROOT:-$CODE_ROOT/work/v5_native_count_stream/targeted_count_chain_20d10c_20260821/$MODEL}"
BANK_PLAN="${BANK_PLAN:-$TARGETED_ROOT/frozen_targeted_count_plan.csv}"
ANCHOR_REGISTRY="${ANCHOR_REGISTRY:-$TARGETED_ROOT/final_transition_registry/selected_anchor_registry.jsonl}"
DISCOVERY="$RUN_ROOT/restoration_discovery"
DISCOVERY_ANALYSIS="$RUN_ROOT/restoration_analysis_discovery"
CONFIRMATION="$RUN_ROOT/restoration_confirmation"
CONFIRMATION_ANALYSIS="$RUN_ROOT/restoration_analysis_confirmation"
COMPLETE="$RUN_ROOT/restoration_complete.json"
LOG="$RUN_ROOT/logs/integrated_mediator_restoration.log"
GEOMETRY="${GEOMETRY:-suffix8}"
if [[ "$GEOMETRY" != "suffix8" && "$GEOMETRY" != "full_span" ]]; then
  echo "unsupported mediator geometry: $GEOMETRY" >&2
  exit 2
fi

test -f "$BANK_PLAN"
test -f "$ANCHOR_REGISTRY"
PLAN_SIGNATURE="$($PYTHON - "$BANK_PLAN" <<'PY'
import csv
import pathlib
import sys

rows = list(csv.DictReader(pathlib.Path(sys.argv[1]).open(encoding="utf-8")))
selected = [row for row in rows if row["condition"] == "selected_bank"]
assert len(selected) == 1, "plan must have exactly one selected bank"
assert "selection_rank" not in rows[0], "selection_rank is prohibited"
print(f"{int(selected[0]['bank_size'])}:{selected[0]['bank_sha256']}")
PY
)"
PLAN_K="${PLAN_SIGNATURE%%:*}"
PLAN_BANK_SHA="${PLAN_SIGNATURE#*:}"
case "$MODEL:$PLAN_SIGNATURE" in
  Qwen3-8B:125:73aaaeb8f314bd867eff7df43e35d84ca52b60058c25a5a2fa7e8ffafc513659|\
  Qwen3-8B:128:ef30a8a083468c6e88cb5b0924403884ad758fedbc743de36dd03ab9bc4a742b|\
  Gemma4-E4B:8:93a174e36bf14938fdea4a147a032e245caef760f9631e193553f9828d8a6874|\
  Gemma4-E4B:6:2a7652c68454a5333f19324ec5517fe8c22b03ef4955088a283229c8576211b1)
    ;;
  *)
    echo "unregistered integrated targeted plan signature: $MODEL:$PLAN_SIGNATURE" >&2
    exit 2
    ;;
esac

mkdir -p "$RUN_ROOT/locks" "$RUN_ROOT/logs"
exec 9>"$RUN_ROOT/locks/integrated_mediator_restoration.lock"
if ! flock -n 9; then
  echo "another mediator-restoration supervisor owns the lock" >&2
  exit 3
fi

timestamp() { date -u +%Y-%m-%dT%H:%M:%SZ; }
run_phase() {
  local role="$1"
  local mechanism="$2"
  local output="$3"
  echo "START $role utc=$(timestamp)" | tee -a "$LOG"
  "$PYTHON" "$RUNNER" integrated-serial-bridge \
    --mechanism-config "$mechanism" \
    --v5-config "$V5_CONFIG" \
    --model "$MODEL" \
    --cache-dir "$CODE_ROOT/work/hf_cache" \
    --device-map auto \
    --torch-dtype bfloat16 \
    --attention-backend sdpa \
    --generations "$GENERATIONS" \
    --seed-role "$role" \
    --cohort one_to_one \
    --row-panel trace_patch \
    --bank-plan "$BANK_PLAN" \
    --anchor-registry "$ANCHOR_REGISTRY" \
    --write-window query_through_trace \
    --bridge-design restoration \
    --geometry "$GEOMETRY" \
    --max-new-tokens 16 \
    --output "$output" 2>&1 | tee -a "$LOG"
  echo "COMPLETE $role utc=$(timestamp)" | tee -a "$LOG"
}

run_analysis() {
  local phase="$1"
  local trials="$2"
  local output="$3"
  "$PYTHON" "$ANALYZER" \
    --trials "$trials" \
    --output "$output" \
    --phase "$phase" 2>&1 | tee -a "$LOG"
}

run_phase development "$DEV_MECHANISM" "$DISCOVERY"
run_analysis discovery "$DISCOVERY" "$DISCOVERY_ANALYSIS"
if ! "$PYTHON" - "$DISCOVERY_ANALYSIS/claim_gates.json" <<'PY'
import json, pathlib, sys
claims = json.loads(pathlib.Path(sys.argv[1]).read_text(encoding="utf-8"))
raise SystemExit(0 if claims["integrated_mediator_restoration_pass"] else 1)
PY
then
  "$PYTHON" - "$DISCOVERY_ANALYSIS/claim_gates.json" "$COMPLETE" "$MODEL" "$GEOMETRY" "$PLAN_K" "$PLAN_BANK_SHA" <<'PY'
import datetime, json, pathlib, sys
claims = json.loads(pathlib.Path(sys.argv[1]).read_text(encoding="utf-8"))
value = {
    "schema_version": "realistic_niah_v5_mediator_restoration_supervisor_v1",
    "model_label": sys.argv[3],
    "mediator_geometry": sys.argv[4],
    "targeted_bank_size": int(sys.argv[5]),
    "targeted_bank_sha256": sys.argv[6],
    "status": "DISCOVERY_GATE_FAIL",
    "completed_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
    "discovery_seed_count": 20,
    "confirmation_opened": False,
    "selection_rank_used": False,
    "integrated_mediator_restoration_pass": False,
    "discovery_claim_gates": claims,
}
path = pathlib.Path(sys.argv[2]); path.parent.mkdir(parents=True, exist_ok=True)
tmp = path.with_suffix(path.suffix + ".tmp")
tmp.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
tmp.replace(path)
PY
  echo "DISCOVERY_NEGATIVE $MODEL; confirmation remains sealed" | tee -a "$LOG"
  exit 0
fi

run_phase confirmation "$CONFIRM_MECHANISM" "$CONFIRMATION"
run_analysis confirmation "$CONFIRMATION" "$CONFIRMATION_ANALYSIS"
"$PYTHON" - "$DISCOVERY_ANALYSIS/audit.json" "$CONFIRMATION_ANALYSIS/audit.json" "$DISCOVERY_ANALYSIS/claim_gates.json" "$CONFIRMATION_ANALYSIS/claim_gates.json" "$COMPLETE" "$MODEL" "$GEOMETRY" "$PLAN_K" "$PLAN_BANK_SHA" <<'PY'
import datetime, json, pathlib, sys
discovery_audit = json.loads(pathlib.Path(sys.argv[1]).read_text(encoding="utf-8"))
confirmation_audit = json.loads(pathlib.Path(sys.argv[2]).read_text(encoding="utf-8"))
discovery_claims = json.loads(pathlib.Path(sys.argv[3]).read_text(encoding="utf-8"))
confirmation_claims = json.loads(pathlib.Path(sys.argv[4]).read_text(encoding="utf-8"))
assert discovery_audit["seed_count"] == 20, discovery_audit
assert confirmation_audit["seed_count"] == 10, confirmation_audit
assert discovery_audit["selection_rank_used"] is False, discovery_audit
assert confirmation_audit["selection_rank_used"] is False, confirmation_audit
passed = bool(confirmation_claims["integrated_mediator_restoration_pass"])
value = {
    "schema_version": "realistic_niah_v5_mediator_restoration_supervisor_v1",
    "model_label": sys.argv[6],
    "mediator_geometry": sys.argv[7],
    "targeted_bank_size": int(sys.argv[8]),
    "targeted_bank_sha256": sys.argv[9],
    "status": "PASS" if passed else "CONFIRMATION_GATE_FAIL",
    "completed_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
    "discovery_seed_count": 20,
    "confirmation_seed_count": 10,
    "selection_rank_used": False,
    "discovery_claim_gates": discovery_claims,
    "confirmation_claim_gates": confirmation_claims,
    "integrated_mediator_restoration_pass": passed,
}
path = pathlib.Path(sys.argv[5]); path.parent.mkdir(parents=True, exist_ok=True)
tmp = path.with_suffix(path.suffix + ".tmp")
tmp.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
tmp.replace(path)
PY

cat "$COMPLETE" | tee -a "$LOG"
echo "FINISHED $MODEL utc=$(timestamp)" | tee -a "$LOG"
