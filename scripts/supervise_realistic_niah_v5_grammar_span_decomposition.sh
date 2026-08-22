#!/usr/bin/env bash
set -euo pipefail

MODEL=${1:?usage: $0 MODEL GPU_INDEX}
GPU_INDEX=${2:?usage: $0 MODEL GPU_INDEX}
ROOT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$ROOT_DIR"

case "$MODEL" in
  Qwen3-8B)
    LAYER=19
    TARGETED_ROOT="$ROOT_DIR/work/v5_native_count_stream/targeted_count_chain_k128_20d10c_20260821_v1/Qwen3-8B"
    ;;
  Gemma4-E4B)
    LAYER=16
    TARGETED_ROOT="$ROOT_DIR/work/v5_native_count_stream/targeted_count_chain_k6_20d10c_20260821_v1/Gemma4-E4B"
    ;;
  *)
    echo "unsupported model: $MODEL" >&2
    exit 2
    ;;
esac

PYTHON="$ROOT_DIR/.venv/bin/python"
BUILDER="$ROOT_DIR/scripts/build_realistic_niah_v5_grammar_span_anchor_panel.py"
RUNNER="$ROOT_DIR/scripts/run_realistic_niah_v5_grammar_span_decomposition.py"
ANALYZER="$ROOT_DIR/scripts/analyze_realistic_niah_v5_grammar_span_decomposition.py"
DEV_CONFIG="$ROOT_DIR/configs/realistic_niah_v5_native_count_stream_dev.json"
CONFIRM_CONFIG="$ROOT_DIR/configs/realistic_niah_v5_native_count_stream_confirmation_v1.json"
V5_CONFIG="$ROOT_DIR/configs/realistic_niah_v5.json"
GENERATIONS="$ROOT_DIR/work/v5_trace_parser_v2/${MODEL}_generations_reparsed.jsonl"
SOURCE_REGISTRY="$TARGETED_ROOT/final_transition_registry/selected_anchor_registry.jsonl"
OUTPUT_ROOT="$ROOT_DIR/work/v5_native_count_stream/grammar_terminal_span_decomposition_k128_k6_20d10c_20260821_v1/$MODEL"
PANEL="$OUTPUT_ROOT/frozen_grammar_span_anchor_panel.jsonl"
PANEL_MANIFEST="$OUTPUT_ROOT/frozen_grammar_span_anchor_manifest.json"
COMPLETE="$OUTPUT_ROOT/grammar_span_decomposition_complete.json"
LOG="$OUTPUT_ROOT/logs/grammar_span_decomposition.log"
mkdir -p "$OUTPUT_ROOT/logs" "$OUTPUT_ROOT/locks"

exec 9>"$OUTPUT_ROOT/locks/grammar_span_decomposition.lock"
if ! flock -n 9; then
  echo "another $MODEL grammar-span supervisor owns the lock" >&2
  exit 3
fi

if [[ ! -f "$SOURCE_REGISTRY" ]]; then
  echo "missing targeted final-transition registry: $SOURCE_REGISTRY" >&2
  exit 4
fi
"$PYTHON" "$BUILDER" \
  --input "$SOURCE_REGISTRY" \
  --output "$PANEL" \
  --manifest "$PANEL_MANIFEST"

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
    --anchor-panel "$PANEL" \
    --anchor-manifest "$PANEL_MANIFEST" \
    --layer "$LAYER" \
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
  "$COMPLETE" "$MODEL" "$LAYER" <<'PY'
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
selected = str(discovery["largest_split_geometry"])
confirmation_primary = confirmation["primary_estimands"]
restore = next(
    row for row in confirmation_primary
    if row["geometry"] == selected and row["contrast"] == "restoration"
)
specificity = next(
    row for row in confirmation_primary
    if row["geometry"] == selected
    and row["contrast"] == "matched_random_specificity"
)
value = {
    "schema_version": "realistic_niah_v5_grammar_span_decomposition_complete_v1",
    "status": "PASS",
    "model_label": sys.argv[6],
    "completed_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
    "discovery_seed_count": 20,
    "confirmation_seed_count": 10,
    "confirmation_ran_unconditionally": True,
    "outcome_blind": True,
    "selection_rank_used": False,
    "source_layer": int(sys.argv[7]),
    "discovery_selected_split_geometry": selected,
    "confirmation_selected_geometry_restoration": restore,
    "confirmation_selected_geometry_matched_random_specificity": specificity,
    "discovery_selected_geometry_confirmation_descriptive_signal": bool(
        restore["mean_effect"] > 0.0 and specificity["mean_effect"] > 0.0
    ),
    "discovery": discovery,
    "confirmation": confirmation,
}
path = pathlib.Path(sys.argv[5])
temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
temporary.replace(path)
print(json.dumps(value, sort_keys=True))
PY
