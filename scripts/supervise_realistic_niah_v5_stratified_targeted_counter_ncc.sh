#!/usr/bin/env bash
set -euo pipefail

MODEL=${1:?usage: $0 MODEL GPU_INDEX}
GPU_INDEX=${2:?usage: $0 MODEL GPU_INDEX}
ROOT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$ROOT_DIR"

case "$MODEL" in
  Qwen3-8B)
    GENERATIONS="$ROOT_DIR/work/v5_supplement_inputs/Qwen3-8B_generations_reparsed.jsonl"
    ;;
  Gemma4-E4B)
    GENERATIONS="$ROOT_DIR/work/v5_supplement_inputs/Gemma4-E4B_generations_reparsed.jsonl"
    ;;
  *)
    echo "unsupported model: $MODEL" >&2
    exit 2
    ;;
esac

PYTHON="$ROOT_DIR/.venv/bin/python"
MECH_DEV="$ROOT_DIR/configs/realistic_niah_v5_native_count_stream_dev.json"
MECH_CONFIRM="$ROOT_DIR/configs/realistic_niah_v5_native_count_stream_confirmation_v1.json"
V5_CONFIG="$ROOT_DIR/configs/realistic_niah_v5.json"
INPUT_ROOT="$ROOT_DIR/work/v5_stratified_ncc_inputs_20260823/$MODEL"
OUTPUT_ROOT="$ROOT_DIR/work/v5_native_count_stream/stratified_ncc_20d10c_20260823_v1/$MODEL"
LOG="$OUTPUT_ROOT/logs/supervisor.log"
LOCK="$OUTPUT_ROOT/locks/supervisor.lock"
COMPLETE="$OUTPUT_ROOT/stratified_ncc_complete.json"
mkdir -p "$OUTPUT_ROOT/logs" "$OUTPUT_ROOT/locks"

exec 9>"$LOCK"
if ! flock -n 9; then
  echo "another $MODEL stratified-NCC supervisor owns the lock" >&2
  exit 3
fi

for path in \
  "$GENERATIONS" \
  "$INPUT_ROOT/rank_after_city_panel.jsonl" \
  "$INPUT_ROOT/rank_after_city_bank_plan.csv" \
  "$INPUT_ROOT/rank_before_city_panel.jsonl" \
  "$INPUT_ROOT/rank_before_city_bank_plan.csv"; do
  test -s "$path" || { echo "missing input: $path" >&2; exit 4; }
done

run_phase() {
  local timing=$1
  local role=$2
  local phase=$3
  local mechanism=$4
  local panel="$INPUT_ROOT/${timing}_panel.jsonl"
  local bank="$INPUT_ROOT/${timing}_bank_plan.csv"
  local output="$OUTPUT_ROOT/$timing/$phase"
  echo "STRATIFIED_NCC_START model=$MODEL timing=$timing phase=$phase utc=$(date -u +%FT%TZ)" | tee -a "$LOG"
  CUDA_VISIBLE_DEVICES="$GPU_INDEX" "$PYTHON" \
    "$ROOT_DIR/scripts/run_realistic_niah_v5_stratified_targeted_counter_ncc.py" \
    --mechanism-config "$mechanism" \
    --v5-config "$V5_CONFIG" \
    --model "$MODEL" \
    --cache-dir "$ROOT_DIR/work/hf_cache" \
    --device-map auto \
    --torch-dtype bfloat16 \
    --attention-backend sdpa \
    --generations "$GENERATIONS" \
    --seed-role "$role" \
    --timing "$timing" \
    --panel "$panel" \
    --bank-plan "$bank" \
    --resume \
    --output "$output" 2>&1 | tee -a "$LOG"
  echo "STRATIFIED_NCC_SEALED model=$MODEL timing=$timing phase=$phase utc=$(date -u +%FT%TZ)" | tee -a "$LOG"
}

for timing in rank_after_city rank_before_city; do
  run_phase "$timing" development discovery "$MECH_DEV"
  run_phase "$timing" confirmation confirmation "$MECH_CONFIRM"
  "$PYTHON" \
    "$ROOT_DIR/scripts/analyze_realistic_niah_v5_stratified_targeted_counter_ncc.py" \
    --discovery "$OUTPUT_ROOT/$timing/discovery" \
    --confirmation "$OUTPUT_ROOT/$timing/confirmation" \
    --timing "$timing" \
    --output "$OUTPUT_ROOT/$timing/analysis" 2>&1 | tee -a "$LOG"
done

"$PYTHON" \
  "$ROOT_DIR/scripts/finalize_realistic_niah_v5_stratified_targeted_counter_ncc.py" \
  --model "$MODEL" \
  --output-root "$OUTPUT_ROOT" \
  --output "$COMPLETE"
echo "STRATIFIED_NCC_COMPLETE model=$MODEL utc=$(date -u +%FT%TZ)" | tee -a "$LOG"
