#!/usr/bin/env bash
set -euo pipefail

GPU_INDEX=${1:?usage: $0 GPU_INDEX}
ROOT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$ROOT_DIR"

MODEL=Gemma4-E4B
PYTHON="$ROOT_DIR/.venv/bin/python"
RUNNER="$ROOT_DIR/scripts/run_realistic_niah_v5_local_terminal_token_state_bridge.py"
ANALYZER="$ROOT_DIR/scripts/analyze_realistic_niah_v5_local_terminal_token_state_bridge.py"
DEV_CONFIG="$ROOT_DIR/configs/realistic_niah_v5_native_count_stream_dev.json"
CONFIRM_CONFIG="$ROOT_DIR/configs/realistic_niah_v5_native_count_stream_confirmation_v1.json"
V5_CONFIG="$ROOT_DIR/configs/realistic_niah_v5.json"
GENERATIONS="$ROOT_DIR/work/v5_trace_parser_v2/${MODEL}_generations_reparsed.jsonl"
ANCHORS="$ROOT_DIR/work/v5_native_count_stream/write_edge_fullspan_k128_k8_20d10c_20260821_v1/Gemma4-E4B/frozen_highest_count_anchor_registry.jsonl"
OUTPUT_ROOT="$ROOT_DIR/work/v5_native_count_stream/local_terminal_token_state_bridge_gemma_20d10c_20260821_v1/$MODEL"
COMPLETE="$OUTPUT_ROOT/local_terminal_token_state_complete.json"
LOG="$OUTPUT_ROOT/logs/local_terminal_token_state.log"
mkdir -p "$OUTPUT_ROOT/logs" "$OUTPUT_ROOT/locks"

exec 9>"$OUTPUT_ROOT/locks/local_terminal_token_state.lock"
if ! flock -n 9; then
  echo "another Gemma local terminal-token-state supervisor owns the lock" >&2
  exit 3
fi

timestamp() { date -u +%Y-%m-%dT%H:%M:%SZ; }
run_phase() {
  local role=$1
  local phase=$2
  local mechanism=$3
  local trials="$OUTPUT_ROOT/${phase}"
  local analysis="$OUTPUT_ROOT/analysis_${phase}"
  echo "START $phase utc=$(timestamp)" | tee -a "$LOG"
  CUDA_VISIBLE_DEVICES="$GPU_INDEX" "$PYTHON" "$RUNNER" \
    --mechanism-config "$mechanism" \
    --v5-config "$V5_CONFIG" \
    --model "$MODEL" \
    --cache-dir "$ROOT_DIR/work/hf_cache" \
    --device-map auto \
    --torch-dtype bfloat16 \
    --attention-backend sdpa \
    --generations "$GENERATIONS" \
    --seed-role "$role" \
    --anchor-registry "$ANCHORS" \
    --layer 16 \
    --max-new-tokens 16 \
    --resume \
    --output "$trials" 2>&1 | tee -a "$LOG"
  "$PYTHON" "$ANALYZER" \
    --input "$trials" \
    --phase "$phase" \
    --bootstrap-samples 10000 \
    --random-seed 20260821 \
    --output "$analysis" 2>&1 | tee -a "$LOG"
  echo "SEALED $phase utc=$(timestamp)" | tee -a "$LOG"
}

run_phase development discovery "$DEV_CONFIG"
run_phase confirmation confirmation "$CONFIRM_CONFIG"

"$PYTHON" - \
  "$OUTPUT_ROOT/analysis_discovery/claim_gates.json" \
  "$OUTPUT_ROOT/analysis_confirmation/claim_gates.json" \
  "$OUTPUT_ROOT/analysis_discovery/audit.json" \
  "$OUTPUT_ROOT/analysis_confirmation/audit.json" \
  "$COMPLETE" <<'PY'
import datetime as dt
import json
import os
import pathlib
import sys

discovery = json.loads(pathlib.Path(sys.argv[1]).read_text())
confirmation = json.loads(pathlib.Path(sys.argv[2]).read_text())
discovery_audit = json.loads(pathlib.Path(sys.argv[3]).read_text())
confirmation_audit = json.loads(pathlib.Path(sys.argv[4]).read_text())
assert discovery_audit["seed_count"] == 20
assert confirmation_audit["seed_count"] == 10
value = {
    "schema_version": "realistic_niah_v5_local_terminal_token_state_complete_v1",
    "status": "PASS",
    "model_label": "Gemma4-E4B",
    "completed_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
    "discovery_seed_count": 20,
    "confirmation_seed_count": 10,
    "confirmation_ran_unconditionally": True,
    "outcome_blind": True,
    "selection_rank_used": False,
    "source_layer": 16,
    "geometry": "terminal_full_item_span_same_position",
    "earlier_trace_tokens_remain_clean": True,
    "claim_scope": "local_terminal_item_token_to_fullspan_state_to_answer",
    "targeted_retrieval_linkage": (
        "modular: pair with frozen Gemma Top-6 city/item retrieval endpoint"
    ),
    "discovery": discovery,
    "confirmation": confirmation,
    "complete_local_terminal_token_state_pass": bool(
        discovery["local_terminal_token_state_mediation_pass"]
        and confirmation["local_terminal_token_state_mediation_pass"]
    ),
}
path = pathlib.Path(sys.argv[5])
temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
temporary.replace(path)
print(json.dumps(value, sort_keys=True))
PY
