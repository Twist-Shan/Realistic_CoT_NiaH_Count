#!/usr/bin/env bash
set -euo pipefail

MODEL="${1:?usage: $0 MODEL}"
CODE_ROOT="${CODE_ROOT:-/home/ubuntu/Realistic_CoT_NiaH_Count}"
PYTHON="${PYTHON:-$CODE_ROOT/.venv/bin/python}"
CACHE_DIR="${CACHE_DIR:-$CODE_ROOT/work/hf_cache}"
PROTOCOL="$CODE_ROOT/configs/realistic_niah_v5_native_loop_chain_v1.json"
MECHANISM="$CODE_ROOT/configs/realistic_niah_v5_native_count_stream_confirmation_v1.json"
V5_CONFIG="$CODE_ROOT/configs/realistic_niah_v5.json"
RUNNER="$CODE_ROOT/scripts/run_realistic_niah_v5_count_stream.py"
ANALYZER="$CODE_ROOT/scripts/analyze_realistic_niah_v5_native_loop.py"
RUN_ROOT="$CODE_ROOT/work/v5_native_count_stream/native_loop_chain_k128_k6_20d10c_20260821_v1/$MODEL"
LOG="$RUN_ROOT/logs/native_loop.log"
COMPLETE="$RUN_ROOT/native_loop_complete.json"

case "$MODEL" in
  Qwen3-8B)
    GENERATIONS="$CODE_ROOT/work/v5_trace_parser_v2/Qwen3-8B_generations_reparsed.jsonl"
    TARGETED_COMPLETE="$CODE_ROOT/work/v5_native_count_stream/targeted_count_chain_k128_20d10c_20260821_v1/Qwen3-8B/targeted_count_complete.json"
    SELECTION="$CODE_ROOT/configs/realistic_niah_v5_qwen_shared_k128_targeted_selection_frozen.json"
    ROUTING="$CODE_ROOT/configs/realistic_niah_v5_qwen_shared_k128_causal_routes_frozen.json"
    BASIS="$CODE_ROOT/work/v5_native_count_stream/representation_20260820/Qwen3-8B/item_end_discovery_basis.npz"
    LAYER=19
    ;;
  Gemma4-E4B)
    GENERATIONS="$CODE_ROOT/work/v5_trace_parser_v2/Gemma4-E4B_generations_reparsed.jsonl"
    TARGETED_COMPLETE="$CODE_ROOT/work/v5_native_count_stream/targeted_count_chain_k6_20d10c_20260821_v1/Gemma4-E4B/targeted_count_complete.json"
    SELECTION="$CODE_ROOT/configs/realistic_niah_v5_gemma_shared_k6_targeted_selection_frozen.json"
    ROUTING="$CODE_ROOT/configs/realistic_niah_v5_gemma_shared_k6_causal_routes_frozen.json"
    BASIS="$CODE_ROOT/work/v5_native_count_stream/representation_20260820/Gemma4-E4B/item_end_discovery_basis.npz"
    LAYER=16
    ;;
  *)
    echo "unsupported model: $MODEL" >&2
    exit 2
    ;;
esac

mkdir -p "$RUN_ROOT/logs" "$RUN_ROOT/locks" "$CACHE_DIR"
exec 9>"$RUN_ROOT/locks/native_loop.lock"
if ! flock -n 9; then
  echo "another native-loop supervisor owns $MODEL" >&2
  exit 75
fi

for path in "$PYTHON" "$PROTOCOL" "$MECHANISM" "$V5_CONFIG" "$RUNNER" "$ANALYZER" "$GENERATIONS" "$SELECTION" "$ROUTING" "$BASIS"; do
  test -e "$path"
done

"$PYTHON" - "$PROTOCOL" "$MODEL" "$SELECTION" "$ROUTING" "$BASIS" "$LAYER" <<'PY'
import hashlib
import json
import pathlib
import sys

protocol_path, model, selection, routing, basis, layer = sys.argv[1:]
protocol = json.loads(pathlib.Path(protocol_path).read_text(encoding="utf-8"))
assert protocol["status"] == "FROZEN_BEFORE_ANY_NATIVE_LOOP_OUTCOME"
assert protocol["seed_contract"]["discovery"] == list(range(1234, 1254))
assert protocol["seed_contract"]["confirmation"] == list(range(1254, 1264))
assert protocol["seed_contract"]["selection_rank_used"] is False
spec = protocol["models"][model]
assert int(spec["p0_state_layer"]) == int(layer)
assert hashlib.sha256(pathlib.Path(basis).read_bytes()).hexdigest() == spec["count_basis_sha256"]
selected = json.loads(pathlib.Path(selection).read_text(encoding="utf-8"))
assert selected["development_selection"]["primary_bank_sha256"] == spec["targeted_bank_sha256"]
routed = json.loads(pathlib.Path(routing).read_text(encoding="utf-8"))
assert routed["head_bank"]["selected_bank_sha256"] == spec["targeted_bank_sha256"]
print(json.dumps({"model": model, "layer": int(layer), "protocol_audit": "PASS"}, sort_keys=True))
PY

run_logged() {
  local label="$1"
  shift
  echo "START $label utc=$(date -u +%FT%TZ)" | tee -a "$LOG"
  "$@" 2>&1 | tee -a "$LOG"
  echo "COMPLETE $label utc=$(date -u +%FT%TZ)" | tee -a "$LOG"
}

echo "WAIT_TARGETED_ENDPOINT $MODEL $TARGETED_COMPLETE" | tee -a "$LOG"
while [[ ! -s "$TARGETED_COMPLETE" ]]; do
  sleep 30
done
echo "TARGETED_ENDPOINT_TERMINAL $MODEL utc=$(date -u +%FT%TZ)" | tee -a "$LOG"

PLAN_D="$RUN_ROOT/plan_discovery_offsets123/native_loop_plan.csv"
PLAN_C="$RUN_ROOT/plan_confirmation_offsets123/native_loop_plan.csv"
P0_D="$RUN_ROOT/p0_discovery_offsets123"
BOUNDARY_D="$RUN_ROOT/boundary_discovery"
ANALYSIS_D="$RUN_ROOT/analysis_discovery"
P0_C="$RUN_ROOT/p0_confirmation_offsets123"
BOUNDARY_C="$RUN_ROOT/boundary_confirmation"
ANALYSIS_C="$RUN_ROOT/analysis_confirmation"

common_rows=(
  --mechanism-config "$MECHANISM"
  --v5-config "$V5_CONFIG"
  --model "$MODEL"
  --cache-dir "$CACHE_DIR"
  --device-map auto
  --torch-dtype bfloat16
  --attention-backend sdpa
  --generations "$GENERATIONS"
  --cohort parser_hit
  --donor-offsets -3 -2 -1 1 2 3
  --random-seed 20260821
)

if [[ ! -s "$PLAN_D" ]]; then
  run_logged plan_discovery \
    "$PYTHON" "$RUNNER" plan-native-loop "${common_rows[@]}" \
      --seed-role development --output "$RUN_ROOT/plan_discovery_offsets123"
fi
if [[ ! -s "$PLAN_C" ]]; then
  run_logged plan_confirmation \
    "$PYTHON" "$RUNNER" plan-native-loop "${common_rows[@]}" \
      --seed-role confirmation --output "$RUN_ROOT/plan_confirmation_offsets123"
fi

run_logged p0_discovery \
  "$PYTHON" "$RUNNER" p0-native-loop "${common_rows[@]}" \
    --seed-role development --plan "$PLAN_D" --basis "$BASIS" --layer "$LAYER" \
    --targeted-selection "$SELECTION" --anchor-routing "$ROUTING" \
    --conditions clean self_patch full_donor_patch count_subspace_transplant \
      norm_matched_orthogonal_patch count_component_removed count_component_restored \
    --max-new-tokens 48 --output "$P0_D"

run_logged boundary_discovery \
  "$PYTHON" "$RUNNER" boundary-native-loop "${common_rows[@]}" \
    --seed-role development --plan "$PLAN_D" --basis "$BASIS" --layer "$LAYER" \
    --conditions clean self_patch full_donor_patch count_subspace_transplant \
      norm_matched_orthogonal_patch \
    --max-new-tokens 64 --output "$BOUNDARY_D"

run_logged analyze_discovery \
  "$PYTHON" "$ANALYZER" --trials "$P0_D" "$BOUNDARY_D" \
    --phase discovery --output "$ANALYSIS_D"

if ! "$PYTHON" - "$ANALYSIS_D/claim_gates.json" <<'PY'
import json
import pathlib
import sys
gates = json.loads(pathlib.Path(sys.argv[1]).read_text(encoding="utf-8"))
raise SystemExit(0 if gates["native_loop_pass"] else 1)
PY
then
  "$PYTHON" - "$TARGETED_COMPLETE" "$ANALYSIS_D/claim_gates.json" "$COMPLETE" <<'PY'
import datetime
import json
import pathlib
import sys
targeted = json.loads(pathlib.Path(sys.argv[1]).read_text(encoding="utf-8"))
discovery = json.loads(pathlib.Path(sys.argv[2]).read_text(encoding="utf-8"))
payload = {
    "status": "DISCOVERY_NEGATIVE",
    "completed_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
    "targeted_endpoint_status": targeted.get("status"),
    "discovery": discovery,
    "confirmation_opened": False,
    "complete_native_loop_pass": False,
}
pathlib.Path(sys.argv[3]).write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
PY
  echo "NATIVE_LOOP_DISCOVERY_NEGATIVE $MODEL" | tee -a "$LOG"
  exit 0
fi

run_logged p0_confirmation \
  "$PYTHON" "$RUNNER" p0-native-loop "${common_rows[@]}" \
    --seed-role confirmation --plan "$PLAN_C" --basis "$BASIS" --layer "$LAYER" \
    --targeted-selection "$SELECTION" --anchor-routing "$ROUTING" \
    --conditions clean self_patch full_donor_patch count_subspace_transplant \
      norm_matched_orthogonal_patch count_component_removed count_component_restored \
    --max-new-tokens 48 --output "$P0_C"

run_logged boundary_confirmation \
  "$PYTHON" "$RUNNER" boundary-native-loop "${common_rows[@]}" \
    --seed-role confirmation --plan "$PLAN_C" --basis "$BASIS" --layer "$LAYER" \
    --conditions clean self_patch full_donor_patch count_subspace_transplant \
      norm_matched_orthogonal_patch \
    --max-new-tokens 64 --output "$BOUNDARY_C"

run_logged analyze_confirmation \
  "$PYTHON" "$ANALYZER" --trials "$P0_C" "$BOUNDARY_C" \
    --phase confirmation --output "$ANALYSIS_C"

"$PYTHON" - "$TARGETED_COMPLETE" "$ANALYSIS_D/claim_gates.json" "$ANALYSIS_C/claim_gates.json" "$COMPLETE" <<'PY'
import datetime
import json
import pathlib
import sys
targeted = json.loads(pathlib.Path(sys.argv[1]).read_text(encoding="utf-8"))
discovery = json.loads(pathlib.Path(sys.argv[2]).read_text(encoding="utf-8"))
confirmation = json.loads(pathlib.Path(sys.argv[3]).read_text(encoding="utf-8"))
passed = bool(
    targeted.get("status") == "PASS"
    and discovery["native_loop_pass"]
    and confirmation["native_loop_pass"]
)
payload = {
    "status": "PASS" if passed else "CONFIRMATION_OR_ENDPOINT_NEGATIVE",
    "completed_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
    "targeted_endpoint_status": targeted.get("status"),
    "discovery": discovery,
    "confirmation": confirmation,
    "complete_native_loop_pass": passed,
}
pathlib.Path(sys.argv[4]).write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
print(json.dumps({"status": payload["status"]}, sort_keys=True))
PY
