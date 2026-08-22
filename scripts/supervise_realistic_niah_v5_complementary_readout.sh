#!/usr/bin/env bash
set -euo pipefail

MODEL="${1:-Qwen3-8B}"
if [[ "$MODEL" != "Qwen3-8B" ]]; then
  echo "complementary readout is frozen only for Qwen3-8B" >&2
  exit 2
fi
CODE_ROOT="${CODE_ROOT:-/home/ubuntu/Realistic_CoT_NiaH_Count}"
PYTHON="${PYTHON:-$CODE_ROOT/.venv/bin/python}"
CACHE_DIR="${CACHE_DIR:-$CODE_ROOT/work/hf_cache}"
RUN_ROOT="${RUN_ROOT:-$CODE_ROOT/work/v5_native_count_stream/complementary_readout_20d10c_20260821/$MODEL}"
RUNNER="$CODE_ROOT/scripts/run_realistic_niah_v5_count_stream.py"
ANALYZER="$CODE_ROOT/scripts/analyze_realistic_niah_v5_complementary_readout.py"
DEV_MECHANISM="$CODE_ROOT/configs/realistic_niah_v5_native_count_stream_dev.json"
CONFIRM_MECHANISM="$CODE_ROOT/configs/realistic_niah_v5_native_count_stream_confirmation_v1.json"
V5_CONFIG="$CODE_ROOT/configs/realistic_niah_v5.json"
GENERATIONS="$CODE_ROOT/work/v5_trace_parser_v2/Qwen3-8B_generations_reparsed.jsonl"
DISCOVERY="$RUN_ROOT/complementary_discovery"
DISCOVERY_ANALYSIS="$RUN_ROOT/complementary_analysis_discovery"
CONFIRMATION="$RUN_ROOT/complementary_confirmation"
CONFIRMATION_ANALYSIS="$RUN_ROOT/complementary_analysis_confirmation"
COMPLETE="$RUN_ROOT/complementary_complete.json"
LOG="$RUN_ROOT/logs/complementary_readout.log"

mkdir -p "$RUN_ROOT/logs" "$RUN_ROOT/locks" "$CACHE_DIR"
exec 9>"$RUN_ROOT/locks/complementary_readout.lock"
if ! flock -n 9; then
  echo "another complementary-readout supervisor owns the lock" >&2
  exit 75
fi

run_logged() {
  local label="$1"
  shift
  echo "START $label utc=$(date -u +%FT%TZ)" | tee -a "$LOG"
  "$@" 2>&1 | tee -a "$LOG"
  echo "COMPLETE $label utc=$(date -u +%FT%TZ)" | tee -a "$LOG"
}

run_phase() {
  local role="$1"
  local mechanism="$2"
  local output="$3"
  run_logged "$role" \
    "$PYTHON" "$RUNNER" complementary-readout \
      --mechanism-config "$mechanism" \
      --v5-config "$V5_CONFIG" \
      --model "$MODEL" \
      --cache-dir "$CACHE_DIR" \
      --device-map auto \
      --torch-dtype bfloat16 \
      --attention-backend sdpa \
      --generations "$GENERATIONS" \
      --seed-role "$role" \
      --cohort one_to_one \
      --row-panel trace_patch \
      --source-layer 19 \
      --relay-layer 26 \
      --geometry suffix8 \
      --max-new-tokens 16 \
      --output "$output"
}

run_analysis() {
  local phase="$1"
  local trials="$2"
  local output="$3"
  run_logged "analyze_$phase" \
    "$PYTHON" "$ANALYZER" \
      --trials "$trials" \
      --output "$output" \
      --phase "$phase"
}

run_phase development "$DEV_MECHANISM" "$DISCOVERY"
run_analysis discovery "$DISCOVERY" "$DISCOVERY_ANALYSIS"

DISCOVERY_PASS="$($PYTHON -c 'import json,sys; print(str(json.load(open(sys.argv[1], encoding="utf-8"))["complementary_readout_pass"]).lower())' "$DISCOVERY_ANALYSIS/claim_gates.json")"
if [[ "$DISCOVERY_PASS" != "true" ]]; then
  "$PYTHON" - "$DISCOVERY_ANALYSIS/claim_gates.json" "$COMPLETE" <<'PY'
import datetime
import json
import pathlib
import sys
claims = json.loads(pathlib.Path(sys.argv[1]).read_text(encoding="utf-8"))
payload = {
    "schema_version": "realistic_niah_v5_complementary_supervisor_v1",
    "model_label": "Qwen3-8B",
    "status": "DISCOVERY_GATE_FAIL",
    "completed_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
    "discovery_seed_count": 20,
    "confirmation_opened": False,
    "complementary_readout_pass": False,
    "discovery_claim_gates": claims,
}
pathlib.Path(sys.argv[2]).write_text(
    json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
)
PY
  echo "DISCOVERY_GATE_FAIL model=$MODEL"
  exit 0
fi

run_phase confirmation "$CONFIRM_MECHANISM" "$CONFIRMATION"
run_analysis confirmation "$CONFIRMATION" "$CONFIRMATION_ANALYSIS"

"$PYTHON" - "$DISCOVERY_ANALYSIS/audit.json" "$CONFIRMATION_ANALYSIS/audit.json" "$DISCOVERY_ANALYSIS/claim_gates.json" "$CONFIRMATION_ANALYSIS/claim_gates.json" "$COMPLETE" <<'PY'
import datetime
import json
import pathlib
import sys
discovery_audit = json.loads(pathlib.Path(sys.argv[1]).read_text(encoding="utf-8"))
confirmation_audit = json.loads(pathlib.Path(sys.argv[2]).read_text(encoding="utf-8"))
discovery_claims = json.loads(pathlib.Path(sys.argv[3]).read_text(encoding="utf-8"))
confirmation_claims = json.loads(pathlib.Path(sys.argv[4]).read_text(encoding="utf-8"))
assert discovery_audit["seed_count"] == 20, discovery_audit
assert confirmation_audit["seed_count"] == 10, confirmation_audit
assert discovery_audit["selection_rank_used"] is False, discovery_audit
assert confirmation_audit["selection_rank_used"] is False, confirmation_audit
passed = bool(confirmation_claims["complementary_readout_pass"])
payload = {
    "schema_version": "realistic_niah_v5_complementary_supervisor_v1",
    "model_label": "Qwen3-8B",
    "status": "PASS" if passed else "CONFIRMATION_GATE_FAIL",
    "completed_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
    "discovery_seed_count": 20,
    "confirmation_seed_count": 10,
    "selection_rank_used": False,
    "discovery_claim_gates": discovery_claims,
    "confirmation_claim_gates": confirmation_claims,
    "complementary_readout_pass": passed,
}
pathlib.Path(sys.argv[5]).write_text(
    json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
)
print(json.dumps(payload, sort_keys=True))
PY

STATUS="$($PYTHON -c 'import json,sys; print(json.load(open(sys.argv[1], encoding="utf-8"))["status"])' "$COMPLETE")"
echo "$STATUS model=$MODEL utc=$(date -u +%FT%TZ)"
