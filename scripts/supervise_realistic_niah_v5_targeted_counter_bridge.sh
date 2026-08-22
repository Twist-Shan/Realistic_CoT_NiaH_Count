#!/usr/bin/env bash
set -euo pipefail

MODEL=${1:?usage: $0 MODEL GPU_INDEX}
GPU_INDEX=${2:?usage: $0 MODEL GPU_INDEX}
ROOT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$ROOT_DIR"

case "$MODEL" in
  Qwen3-8B)
    SOURCE_LAYER=19
    BANK_SIZE=128
    BANK_SHA=ef30a8a083468c6e88cb5b0924403884ad758fedbc743de36dd03ab9bc4a742b
    TARGETED_ROOT="$ROOT_DIR/work/v5_native_count_stream/targeted_count_chain_k128_20d10c_20260821_v1/$MODEL"
    ;;
  Gemma4-E4B)
    SOURCE_LAYER=16
    BANK_SIZE=6
    BANK_SHA=2a7652c68454a5333f19324ec5517fe8c22b03ef4955088a283229c8576211b1
    TARGETED_ROOT="$ROOT_DIR/work/v5_native_count_stream/targeted_count_chain_k6_20d10c_20260821_v1/$MODEL"
    ;;
  *)
    echo "unsupported model: $MODEL" >&2
    exit 2
    ;;
esac

PYTHON="$ROOT_DIR/.venv/bin/python"
RUNNER="$ROOT_DIR/scripts/run_realistic_niah_v5_generated_suffix_state_bridge.py"
ANALYZER="$ROOT_DIR/scripts/analyze_realistic_niah_v5_targeted_counter_bridge.py"
DEV_CONFIG="$ROOT_DIR/configs/realistic_niah_v5_native_count_stream_dev.json"
CONFIRM_CONFIG="$ROOT_DIR/configs/realistic_niah_v5_native_count_stream_confirmation_v1.json"
V5_CONFIG="$ROOT_DIR/configs/realistic_niah_v5.json"
GENERATIONS="$ROOT_DIR/work/v5_trace_parser_v2/${MODEL}_generations_reparsed.jsonl"
BANK_PLAN="$TARGETED_ROOT/frozen_targeted_count_plan.csv"
TARGETED_REGISTRY="$TARGETED_ROOT/final_transition_registry/selected_anchor_registry.jsonl"
GRAMMAR_ROOT="$ROOT_DIR/work/v5_native_count_stream/grammar_terminal_span_decomposition_k128_k6_20d10c_20260821_v1/$MODEL"
ANCHORS="$GRAMMAR_ROOT/frozen_grammar_span_anchor_panel.jsonl"
ANCHOR_MANIFEST="$GRAMMAR_ROOT/frozen_grammar_span_anchor_manifest.json"
OUTPUT_ROOT="$ROOT_DIR/work/v5_native_count_stream/targeted_counter_bridge_k128_k6_20d10c_20260821_v1/$MODEL"
COMPLETE="$OUTPUT_ROOT/targeted_counter_complete.json"
LOG="$OUTPUT_ROOT/logs/targeted_counter.log"
mkdir -p "$OUTPUT_ROOT/logs" "$OUTPUT_ROOT/locks"

exec 9>"$OUTPUT_ROOT/locks/targeted_counter.lock"
if ! flock -n 9; then
  echo "another $MODEL targeted-counter supervisor owns the lock" >&2
  exit 3
fi

"$PYTHON" - "$BANK_PLAN" "$TARGETED_REGISTRY" "$ANCHORS" "$ANCHOR_MANIFEST" "$MODEL" "$BANK_SIZE" "$BANK_SHA" <<'PY'
import hashlib, json, pathlib, pandas as pd, sys

bank, targeted, anchors, manifest = map(pathlib.Path, sys.argv[1:5])
model, bank_size, bank_sha = sys.argv[5], int(sys.argv[6]), sys.argv[7]
frame = pd.read_csv(bank)
assert "selection_rank" not in frame.columns
frame = frame[frame["model_label"].astype(str).eq(model)]
assert frame["condition"].astype(str).value_counts().to_dict() == {
    "layer_matched_random": 3,
    "selected_bank": 1,
}
selected = frame[frame["condition"].astype(str).eq("selected_bank")]
assert len(selected) == 1
assert int(selected.iloc[0]["bank_size"]) == bank_size
assert str(selected.iloc[0]["bank_sha256"]) == bank_sha
anchor_rows = [json.loads(line) for line in anchors.read_text().splitlines() if line.strip()]
targeted_rows = [json.loads(line) for line in targeted.read_text().splitlines() if line.strip()]
panel = json.loads(manifest.read_text())
assert len(anchor_rows) == 30 and len({int(row["seed"]) for row in anchor_rows}) == 30
assert panel["outcome_blind"] is True and panel["selection_rank_used"] is False
assert panel["timing_counts_by_phase"] == {
    "development": {"rank_after_city": 10, "rank_before_city": 10},
    "confirmation": {"rank_after_city": 5, "rank_before_city": 5},
}
assert not any("selection_rank" in row for row in anchor_rows + targeted_rows)
assert {row["request_id"] for row in anchor_rows} <= {
    row["request_id"] for row in targeted_rows
}
print(json.dumps({
    "status": "PASS",
    "model_label": model,
    "bank_size": bank_size,
    "bank_sha256": bank_sha,
    "anchor_panel_sha256": hashlib.sha256(anchors.read_bytes()).hexdigest(),
}, sort_keys=True))
PY

timestamp() { date -u +%Y-%m-%dT%H:%M:%SZ; }
run_phase() {
  local role=$1
  local phase=$2
  local mechanism=$3
  local trials="$OUTPUT_ROOT/$phase"
  local analysis="$OUTPUT_ROOT/analysis_${phase}"
  echo "START model=$MODEL phase=$phase utc=$(timestamp)" | tee -a "$LOG"
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
    --source-layer "$SOURCE_LAYER" \
    --state-patch-geometry grammar_counter_carrier \
    --include-matched-position-control \
    --selection-rule frozen_balanced_grammar_specific_counter_carrier \
    --panel-id grammar_balanced_counter_carrier \
    --max-new-tokens 16 \
    --resume \
    --output "$trials" 2>&1 | tee -a "$LOG"
  "$PYTHON" "$ANALYZER" \
    --input "$trials" \
    --phase "$phase" \
    --bootstrap-samples 10000 \
    --random-seed 20260821 \
    --output "$analysis" 2>&1 | tee -a "$LOG"
  echo "SEALED model=$MODEL phase=$phase utc=$(timestamp)" | tee -a "$LOG"
}

# Confirmation is unconditional so the fixed 20d/10c contract is never made
# contingent on a discovery outcome.
run_phase development discovery "$DEV_CONFIG"
run_phase confirmation confirmation "$CONFIRM_CONFIG"

"$PYTHON" - \
  "$OUTPUT_ROOT/analysis_discovery/claim_gates.json" \
  "$OUTPUT_ROOT/analysis_confirmation/claim_gates.json" \
  "$OUTPUT_ROOT/analysis_discovery/audit.json" \
  "$OUTPUT_ROOT/analysis_confirmation/audit.json" \
  "$COMPLETE" "$MODEL" "$BANK_SIZE" "$BANK_SHA" "$SOURCE_LAYER" <<'PY'
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
assert discovery_audit["conditions_per_seed"] == 11
assert confirmation_audit["conditions_per_seed"] == 11
value = {
    "schema_version": "realistic_niah_v5_targeted_counter_bridge_complete_v1",
    "status": "PASS",
    "model_label": sys.argv[6],
    "completed_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
    "discovery_seed_count": 20,
    "confirmation_seed_count": 10,
    "confirmation_ran_unconditionally": True,
    "outcome_blind": True,
    "selection_rank_used": False,
    "targeted_bank_size": int(sys.argv[7]),
    "targeted_bank_sha256": sys.argv[8],
    "source_layer": int(sys.argv[9]),
    "state_patch_geometry": "grammar_counter_carrier",
    "matched_position_control": True,
    "discovery": discovery,
    "confirmation": confirmation,
    "complete_directional_signal_pass": bool(
        discovery["targeted_counter_directional_signal_pass"]
        and confirmation["targeted_counter_directional_signal_pass"]
    ),
    "complete_strong_gate_pass": bool(
        discovery["targeted_counter_strong_gate_pass"]
        and confirmation["targeted_counter_strong_gate_pass"]
    ),
}
path = pathlib.Path(sys.argv[5])
temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
temporary.replace(path)
print(json.dumps(value, sort_keys=True))
PY

cat "$COMPLETE" | tee -a "$LOG"
