#!/usr/bin/env bash
set -euo pipefail

MODEL=${1:?usage: $0 MODEL GPU_INDEX}
GPU_INDEX=${2:?usage: $0 MODEL GPU_INDEX}
ROOT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$ROOT_DIR"

case "$MODEL" in
  Qwen3-8B)
    SEED=1254
    COUNT=10
    SOURCE_LAYER=19
    REQUEST_ID='Qwen3-8B/native_thinking/v5/V4_4_T10000_N10_seed1254'
    ;;
  Gemma4-E4B)
    SEED=1258
    COUNT=10
    SOURCE_LAYER=16
    REQUEST_ID='Gemma4-E4B/native_thinking/v5/V4_4_T10000_N10_seed1258'
    ;;
  *)
    echo "unsupported model: $MODEL" >&2
    exit 2
    ;;
esac

PYTHON="$ROOT_DIR/.venv/bin/python"
RUNNER="$ROOT_DIR/scripts/run_realistic_niah_v5_single_seed_walkthrough.py"
ANALYZER="$ROOT_DIR/scripts/analyze_realistic_niah_v5_single_seed_walkthrough.py"
MECHANISM="$ROOT_DIR/configs/realistic_niah_v5_native_count_stream_confirmation_v1.json"
WALKTHROUGH_CONFIG="$ROOT_DIR/configs/realistic_niah_v5_single_seed_walkthrough_v2.json"
V5_CONFIG="$ROOT_DIR/configs/realistic_niah_v5.json"
GENERATIONS="$ROOT_DIR/work/v5_trace_parser_v2/${MODEL}_generations_reparsed.jsonl"
OUTPUT_ROOT="$ROOT_DIR/work/v5_native_count_stream/single_seed_counter_walkthrough_20260822_v2/$MODEL"
LOG="$OUTPUT_ROOT/walkthrough.log"
mkdir -p "$OUTPUT_ROOT" "$OUTPUT_ROOT/locks"

exec 9>"$OUTPUT_ROOT/locks/walkthrough.lock"
if ! flock -n 9; then
  echo "another $MODEL walkthrough supervisor owns the lock" >&2
  exit 3
fi

CUDA_VISIBLE_DEVICES="$GPU_INDEX" "$PYTHON" "$RUNNER" \
  --mechanism-config "$MECHANISM" \
  --walkthrough-config "$WALKTHROUGH_CONFIG" \
  --v5-config "$V5_CONFIG" \
  --model "$MODEL" \
  --cache-dir "$ROOT_DIR/work/hf_cache" \
  --device-map auto \
  --torch-dtype bfloat16 \
  --attention-backend sdpa \
  --generations "$GENERATIONS" \
  --request-id "$REQUEST_ID" \
  --seed "$SEED" \
  --expected-count "$COUNT" \
  --source-layer "$SOURCE_LAYER" \
  --max-new-tokens 16 \
  --output "$OUTPUT_ROOT" 2>&1 | tee "$LOG"

"$PYTHON" "$ANALYZER" \
  --input "$OUTPUT_ROOT/walkthrough_rows.jsonl" \
  --output "$OUTPUT_ROOT/analysis" 2>&1 | tee -a "$LOG"

test -f "$OUTPUT_ROOT/analysis/walkthrough_complete.json"
echo "PASS $MODEL single-seed walkthrough"
