#!/usr/bin/env bash
set -euo pipefail

GPU_INDEX=${1:?usage: $0 GPU_INDEX}
ROOT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$ROOT_DIR"

MODEL=Gemma4-E4B
PYTHON="$ROOT_DIR/.venv/bin/python"
RUNNER="$ROOT_DIR/scripts/run_realistic_niah_v5_generated_suffix_state_bridge.py"
ANALYZER="$ROOT_DIR/scripts/analyze_realistic_niah_v5_generated_suffix_state_bridge.py"
DEV_CONFIG="$ROOT_DIR/configs/realistic_niah_v5_native_count_stream_dev.json"
CONFIRM_CONFIG="$ROOT_DIR/configs/realistic_niah_v5_native_count_stream_confirmation_v1.json"
V5_CONFIG="$ROOT_DIR/configs/realistic_niah_v5.json"
GENERATIONS="$ROOT_DIR/work/v5_trace_parser_v2/${MODEL}_generations_reparsed.jsonl"
TARGETED_ROOT="$ROOT_DIR/work/v5_native_count_stream/targeted_count_chain_k6_20d10c_20260821_v1/$MODEL"
BANK_PLAN="$TARGETED_ROOT/frozen_targeted_count_plan.csv"
TARGETED_REGISTRY="$TARGETED_ROOT/final_transition_registry/selected_anchor_registry.jsonl"
ANCHORS="$ROOT_DIR/work/v5_native_count_stream/write_edge_fullspan_k128_k8_20d10c_20260821_v1/$MODEL/frozen_highest_count_anchor_registry.jsonl"
OUTPUT_ROOT="$ROOT_DIR/work/v5_native_count_stream/generated_suffix_state_bridge_top6_20d10c_20260821_v2/$MODEL"
COMPLETE="$OUTPUT_ROOT/generated_suffix_state_complete.json"
LOG="$OUTPUT_ROOT/logs/generated_suffix_state.log"
mkdir -p "$OUTPUT_ROOT/logs" "$OUTPUT_ROOT/locks"

exec 9>"$OUTPUT_ROOT/locks/generated_suffix_state.lock"
if ! flock -n 9; then
  echo "another Gemma generated-suffix supervisor owns the lock" >&2
  exit 3
fi

"$PYTHON" - "$BANK_PLAN" "$ANCHORS" "$TARGETED_REGISTRY" <<'PY'
import hashlib, json, pathlib, pandas as pd, sys
bank, anchors, targeted = map(pathlib.Path, sys.argv[1:])
frame = pd.read_csv(bank)
assert "selection_rank" not in frame.columns
selected = frame[frame["condition"].astype(str).eq("selected_bank")]
assert len(selected) == 1
assert int(selected.iloc[0]["bank_size"]) == 6
assert str(selected.iloc[0]["bank_sha256"]) == "2a7652c68454a5333f19324ec5517fe8c22b03ef4955088a283229c8576211b1"
assert frame["condition"].astype(str).value_counts().to_dict() == {
    "layer_matched_random": 3, "selected_bank": 1
}
anchor_rows = [json.loads(line) for line in anchors.read_text().splitlines() if line.strip()]
targeted_rows = [json.loads(line) for line in targeted.read_text().splitlines() if line.strip()]
assert len(anchor_rows) == 30
assert len({int(row["seed"]) for row in anchor_rows}) == 30
assert not any("selection_rank" in row for row in anchor_rows + targeted_rows)
assert {row["request_id"] for row in anchor_rows} <= {
    row["request_id"] for row in targeted_rows
}
print(json.dumps({
    "status": "PASS",
    "anchor_rows": len(anchor_rows),
    "selected_bank_sha256": str(selected.iloc[0]["bank_sha256"]),
    "bank_plan_sha256": hashlib.sha256(bank.read_bytes()).hexdigest(),
}, sort_keys=True))
PY

timestamp() { date -u +%Y-%m-%dT%H:%M:%SZ; }
run_phase() {
  local role=$1
  local phase=$2
  local mechanism=$3
  local trials="$OUTPUT_ROOT/$phase"
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
    --targeted-registry "$TARGETED_REGISTRY" \
    --bank-plan "$BANK_PLAN" \
    --source-layer 16 \
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

# Confirmation remains unconditional under the fixed 20d/10c protocol so all
# magnitudes are retained even when a formal composite gate is negative.
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
assert not discovery_audit["teacher_forced_terminal_suffix"]
assert not confirmation_audit["teacher_forced_terminal_suffix"]
value = {
    "schema_version": "realistic_niah_v5_generated_suffix_state_complete_v1",
    "status": "PASS",
    "model_label": "Gemma4-E4B",
    "completed_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
    "discovery_seed_count": 20,
    "confirmation_seed_count": 10,
    "confirmation_ran_unconditionally": True,
    "outcome_blind": True,
    "selection_rank_used": False,
    "targeted_bank_size": 6,
    "targeted_bank_sha256": "2a7652c68454a5333f19324ec5517fe8c22b03ef4955088a283229c8576211b1",
    "source_layer": 16,
    "state_patch_layers": list(range(16, 42)),
    "terminal_suffix_free_running": True,
    "fixed_token_budget_alignment": True,
    "post_terminal_suffix_teacher_forced": True,
    "discovery": discovery,
    "confirmation": confirmation,
    "complete_generated_suffix_state_bridge_pass": bool(
        discovery["generated_suffix_state_bridge_pass"]
        and confirmation["generated_suffix_state_bridge_pass"]
    ),
}
path = pathlib.Path(sys.argv[5])
temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
temporary.replace(path)
print(json.dumps(value, sort_keys=True))
PY

cat "$COMPLETE" | tee -a "$LOG"
