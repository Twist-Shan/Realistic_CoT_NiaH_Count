#!/usr/bin/env bash
set -euo pipefail

GPU_INDEX=${1:?usage: $0 GPU_INDEX}
ROOT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$ROOT_DIR"

MODEL=Qwen3-8B
BANK_K=128
BANK_SHA=ef30a8a083468c6e88cb5b0924403884ad758fedbc743de36dd03ab9bc4a742b
TARGETED_ROOT="$ROOT_DIR/work/v5_native_count_stream/targeted_count_chain_k128_20d10c_20260821_v1/Qwen3-8B"
PYTHON="$ROOT_DIR/.venv/bin/python"
RUNNER="$ROOT_DIR/scripts/run_realistic_niah_v5_count_stream.py"
ANALYZER="$ROOT_DIR/scripts/analyze_realistic_niah_v5_integrated_mediator_restoration.py"
SUBSET_BUILDER="$ROOT_DIR/scripts/build_realistic_niah_v5_write_edge_midcount_anchor_subset.py"
DEV_CONFIG="$ROOT_DIR/configs/realistic_niah_v5_native_count_stream_dev.json"
CONFIRM_CONFIG="$ROOT_DIR/configs/realistic_niah_v5_native_count_stream_confirmation_v1.json"
V5_CONFIG="$ROOT_DIR/configs/realistic_niah_v5.json"
GENERATIONS="$ROOT_DIR/work/v5_trace_parser_v2/${MODEL}_generations_reparsed.jsonl"
BANK_PLAN="$TARGETED_ROOT/frozen_targeted_count_plan.csv"
SOURCE_REGISTRY="$TARGETED_ROOT/final_transition_registry/selected_anchor_registry.jsonl"
OUTPUT_ROOT="$ROOT_DIR/work/v5_native_count_stream/write_edge_fullspan_k128_midcount58_20d10c_20260821_v2/$MODEL"
ANCHOR_SUBSET="$OUTPUT_ROOT/frozen_midcount_geometry_eligible_anchor_registry.jsonl"
ANCHOR_AUDIT="$OUTPUT_ROOT/frozen_midcount_geometry_eligible_anchor_registry.audit.json"
COMPLETE="$OUTPUT_ROOT/write_edge_complete.json"
LOG="$OUTPUT_ROOT/logs/write_edge_fullspan_midcount.log"
mkdir -p "$OUTPUT_ROOT/logs" "$OUTPUT_ROOT/locks"

exec 9>"$OUTPUT_ROOT/locks/write_edge_fullspan_midcount.lock"
if ! flock -n 9; then
  echo "another Qwen mid-count write-edge supervisor owns the lock" >&2
  exit 3
fi

"$PYTHON" "$SUBSET_BUILDER" \
  --input "$SOURCE_REGISTRY" \
  --generations "$GENERATIONS" \
  --model "$MODEL" \
  --cache-dir "$ROOT_DIR/work/hf_cache" \
  --output "$ANCHOR_SUBSET" | tee "$OUTPUT_ROOT/anchor_subset_audit.log"

timestamp() { date -u +%Y-%m-%dT%H:%M:%SZ; }
run_phase() {
  local role=$1
  local phase=$2
  local mechanism=$3
  local trials="$OUTPUT_ROOT/write_${phase}"
  local analysis="$OUTPUT_ROOT/analysis_${phase}"
  echo "START $phase utc=$(timestamp)" | tee -a "$LOG"
  CUDA_VISIBLE_DEVICES="$GPU_INDEX" "$PYTHON" "$RUNNER" \
    integrated-serial-bridge \
    --mechanism-config "$mechanism" \
    --v5-config "$V5_CONFIG" \
    --model "$MODEL" \
    --cache-dir "$ROOT_DIR/work/hf_cache" \
    --device-map auto \
    --torch-dtype bfloat16 \
    --attention-backend sdpa \
    --generations "$GENERATIONS" \
    --seed-role "$role" \
    --cohort one_to_one \
    --row-panel trace_patch \
    --bank-plan "$BANK_PLAN" \
    --anchor-registry "$ANCHOR_SUBSET" \
    --write-window query_through_trace \
    --bridge-design restoration \
    --geometry full_span \
    --max-new-tokens 16 \
    --output "$trials" 2>&1 | tee -a "$LOG"
  "$PYTHON" "$ANALYZER" \
    --trials "$trials" \
    --output "$analysis" \
    --phase "$phase" \
    --bootstrap-samples 10000 \
    --random-seed 20260821 2>&1 | tee -a "$LOG"
  echo "SEALED $phase utc=$(timestamp)" | tee -a "$LOG"
}

run_phase development discovery "$DEV_CONFIG"
run_phase confirmation confirmation "$CONFIRM_CONFIG"

"$PYTHON" - \
  "$OUTPUT_ROOT/analysis_discovery/claim_gates.json" \
  "$OUTPUT_ROOT/analysis_confirmation/claim_gates.json" \
  "$OUTPUT_ROOT/analysis_discovery/audit.json" \
  "$OUTPUT_ROOT/analysis_confirmation/audit.json" \
  "$ANCHOR_AUDIT" "$COMPLETE" "$BANK_K" "$BANK_SHA" <<'PY'
import datetime as dt
import json
import os
import pathlib
import sys

discovery = json.loads(pathlib.Path(sys.argv[1]).read_text())
confirmation = json.loads(pathlib.Path(sys.argv[2]).read_text())
discovery_audit = json.loads(pathlib.Path(sys.argv[3]).read_text())
confirmation_audit = json.loads(pathlib.Path(sys.argv[4]).read_text())
anchor_audit = json.loads(pathlib.Path(sys.argv[5]).read_text())
assert discovery_audit["seed_count"] == 20
assert confirmation_audit["seed_count"] == 10
assert anchor_audit["fixed_count_band"] == [5, 8]
assert anchor_audit["row_count"] == 30
value = {
    "schema_version": "realistic_niah_v5_write_edge_midcount_complete_v1",
    "status": "PASS",
    "model_label": "Qwen3-8B",
    "completed_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
    "targeted_bank_size": int(sys.argv[7]),
    "targeted_bank_sha256": sys.argv[8],
    "fixed_count_band": [5, 8],
    "followup_status": "secondary_after_highest_count_restoration_negative",
    "geometry": "full_span",
    "write_window": "query_through_trace",
    "bridge_design": "restoration",
    "discovery_seed_count": 20,
    "confirmation_seed_count": 10,
    "confirmation_ran_unconditionally": True,
    "outcome_blind": True,
    "selection_rank_used": False,
    "anchor_subset_audit": anchor_audit,
    "discovery": discovery,
    "confirmation": confirmation,
    "complete_write_edge_formal_pass": bool(
        discovery["integrated_mediator_restoration_pass"]
        and confirmation["integrated_mediator_restoration_pass"]
    ),
}
path = pathlib.Path(sys.argv[6])
temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
temporary.replace(path)
print(json.dumps(value, sort_keys=True))
PY
