#!/usr/bin/env bash
set -euo pipefail

MODEL=${1:?usage: $0 MODEL GPU_INDEX}
GPU_INDEX=${2:?usage: $0 MODEL GPU_INDEX}
ROOT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$ROOT_DIR"

case "$MODEL" in
  Qwen3-8B)
    LAYER=19
    SELECTION="$ROOT_DIR/configs/realistic_niah_v5_qwen_shared_k128_targeted_selection_frozen.json"
    ROUTING="$ROOT_DIR/configs/realistic_niah_v5_qwen_shared_k128_causal_routes_frozen.json"
    HEAD_PLAN="$ROOT_DIR/work/v5_native_count_stream/query_mediation_ladder_k128_20d10c_20260821_v1/Qwen3-8B/head_plan/query_mediation_head_plan.json"
    ;;
  Gemma4-E4B)
    LAYER=16
    SELECTION="$ROOT_DIR/configs/realistic_niah_v5_gemma_shared_k6_targeted_selection_frozen.json"
    ROUTING="$ROOT_DIR/configs/realistic_niah_v5_gemma_shared_k6_causal_routes_frozen.json"
    HEAD_PLAN="$ROOT_DIR/work/v5_native_count_stream/query_mediation_ladder_k6_20d10c_20260821_v1/Gemma4-E4B/head_plan/query_mediation_head_plan.json"
    ;;
  *)
    echo "unsupported model: $MODEL" >&2
    exit 2
    ;;
esac

PYTHON="$ROOT_DIR/.venv/bin/python"
CONFIG="$ROOT_DIR/configs/realistic_niah_v5_native_count_stream_confirmation_v1.json"
V5_CONFIG="$ROOT_DIR/configs/realistic_niah_v5.json"
GENERATIONS="$ROOT_DIR/work/v5_trace_parser_v2/${MODEL}_generations_reparsed.jsonl"
OUTPUT_ROOT="$ROOT_DIR/work/v5_native_count_stream/html_aligned_local_head_span_20d10c_20260821_v1/$MODEL"
TERMINAL_COMPLETE="$ROOT_DIR/work/v5_native_count_stream/html_aligned_terminal_fullspan_20d10c_20260821_v1/$MODEL/html_aligned_terminal_complete.json"
mkdir -p "$OUTPUT_ROOT/logs"

while [[ ! -f "$TERMINAL_COMPLETE" ]]; do
  sleep 30
done

run_phase() {
  local role=$1
  local phase=$2
  local trial_root="$OUTPUT_ROOT/local_${phase}"
  local analysis_root="$OUTPUT_ROOT/analysis_${phase}"
  CUDA_VISIBLE_DEVICES="$GPU_INDEX" "$PYTHON" \
    "$ROOT_DIR/scripts/run_realistic_niah_v5_count_stream.py" \
    html-aligned-local-serial \
    --mechanism-config "$CONFIG" \
    --v5-config "$V5_CONFIG" \
    --model "$MODEL" \
    --cache-dir "$ROOT_DIR/work/hf_cache" \
    --device-map auto \
    --torch-dtype bfloat16 \
    --attention-backend sdpa \
    --generations "$GENERATIONS" \
    --cohort one_to_one \
    --seed-role "$role" \
    --layer "$LAYER" \
    --targeted-selection "$SELECTION" \
    --anchor-routing "$ROUTING" \
    --head-plan "$HEAD_PLAN" \
    --head-token-geometry query_plus_full_path \
    --random-seed 20260821 \
    --skip-greedy \
    --output "$trial_root"
  "$PYTHON" \
    "$ROOT_DIR/scripts/analyze_realistic_niah_v5_html_aligned_local_head_span.py" \
    --input "$trial_root" \
    --phase "$phase" \
    --bootstrap-samples 10000 \
    --random-seed 20260821 \
    --output "$analysis_root"
}

run_phase development discovery
run_phase confirmation confirmation

"$PYTHON" - "$OUTPUT_ROOT" "$MODEL" <<'PY'
import datetime as dt
import json
import os
import pathlib
import sys

root = pathlib.Path(sys.argv[1])
model = sys.argv[2]
discovery = json.loads((root / "analysis_discovery" / "claim_gates.json").read_text())
confirmation = json.loads((root / "analysis_confirmation" / "claim_gates.json").read_text())
value = {
    "schema_version": "realistic_niah_v5_html_local_head_span_complete_v1",
    "model_label": model,
    "status": "PASS",
    "completed_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
    "discovery": discovery,
    "confirmation": confirmation,
    "complete_html_local_head_span_pass": bool(
        discovery["complete_local_head_span_pass"]
        and confirmation["complete_local_head_span_pass"]
    ),
}
path = root / "html_local_head_span_complete.json"
temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
temporary.replace(path)
PY
