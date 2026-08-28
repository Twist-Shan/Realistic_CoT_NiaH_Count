#!/usr/bin/env bash
set -euo pipefail

MODEL=${1:?usage: $0 MODEL GPU_INDEX}
GPU_INDEX=${2:?usage: $0 MODEL GPU_INDEX}
ROOT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$ROOT_DIR"
PYTHON="$ROOT_DIR/.venv/bin/python"
GENERATIONS="$ROOT_DIR/work/v5_supplement_inputs/${MODEL}_generations_reparsed.jsonl"
DISCOVERY_SEEDS=(1234 1235 1236 1237 1238 1239 1240 1241 1242 1243 1244 1245 1246 1247 1248 1249 1250 1251 1252 1253)
CONFIRMATION_SEEDS=(1254 1255 1256 1257 1258 1259 1260 1261 1262 1263)

case "$MODEL" in
  Qwen3-8B)
    OUTPUT_VERSION=indexed_old_html_counter_single_layer_20d10c_20260823_v1
    DISCOVERY_LAYERS=(0 4 8 12 16 20 24 28 32)
    ;;
  Gemma4-E4B)
    OUTPUT_VERSION=indexed_old_html_counter_single_layer_20d10c_20260823_v1
    DISCOVERY_LAYERS=(0 4 8 12 16 20 24 28 32 36 40)
    ;;
  *)
    echo "unsupported model: $MODEL" >&2
    exit 2
    ;;
esac

OUTPUT_ROOT="$ROOT_DIR/work/v5_native_count_stream/$OUTPUT_VERSION/$MODEL"
LOG="$OUTPUT_ROOT/logs/supervisor.log"
LOCK="$OUTPUT_ROOT/locks/supervisor.lock"
COMPLETE="$OUTPUT_ROOT/indexed_counter_patch_complete.json"

mkdir -p "$OUTPUT_ROOT/logs" "$OUTPUT_ROOT/locks"
exec 9>"$LOCK"
flock -n 9 || { echo "another $MODEL indexed-patch supervisor owns the lock" >&2; exit 3; }
test -x "$PYTHON"
test -s "$GENERATIONS"
if test -s "$COMPLETE"; then
  echo "already complete: $COMPLETE"
  exit 0
fi

run_patch() {
  local phase=$1
  local output=$2
  shift 2
  local layers=("$@")
  local seeds
  if [[ "$phase" == discovery ]]; then
    seeds=("${DISCOVERY_SEEDS[@]}")
  else
    seeds=("${CONFIRMATION_SEEDS[@]}")
  fi
  CUDA_VISIBLE_DEVICES="$GPU_INDEX" "$PYTHON" \
    scripts/run_realistic_niah_v5_indexed_counter_early_stop_patch.py \
    --model "$MODEL" \
    --cache-dir "$ROOT_DIR/work/hf_cache" \
    --device-map auto \
    --torch-dtype bfloat16 \
    --attention-backend sdpa \
    --generations "$GENERATIONS" \
    --phase "$phase" \
    --expected-seeds "${seeds[@]}" \
    --source-layers "${layers[@]}" \
    --resume \
    --output "$output" 2>&1 | tee -a "$LOG"
}

if ! test -s "$OUTPUT_ROOT/discovery/manifest.json"; then
  echo "INDEXED_PATCH_DISCOVERY_START model=$MODEL utc=$(date -u +%FT%TZ)" | tee -a "$LOG"
  run_patch discovery "$OUTPUT_ROOT/discovery" "${DISCOVERY_LAYERS[@]}"
fi
if ! test -s "$OUTPUT_ROOT/analysis_discovery/claim_gates.json"; then
  "$PYTHON" scripts/analyze_realistic_niah_v5_indexed_counter_early_stop_patch.py \
    --input "$OUTPUT_ROOT/discovery" \
    --phase discovery \
    --expected-seeds "${DISCOVERY_SEEDS[@]}" \
    --output "$OUTPUT_ROOT/analysis_discovery" 2>&1 | tee -a "$LOG"
fi

SELECTED_LAYER=$("$PYTHON" -c \
  'import json,sys; print(json.load(open(sys.argv[1]))["selected_layer"])' \
  "$OUTPUT_ROOT/analysis_discovery/claim_gates.json")
if ! test -s "$OUTPUT_ROOT/confirmation/manifest.json"; then
  echo "INDEXED_PATCH_CONFIRMATION_START model=$MODEL layer=$SELECTED_LAYER utc=$(date -u +%FT%TZ)" | tee -a "$LOG"
  run_patch confirmation "$OUTPUT_ROOT/confirmation" "$SELECTED_LAYER"
fi
if ! test -s "$OUTPUT_ROOT/analysis_confirmation/claim_gates.json"; then
  "$PYTHON" scripts/analyze_realistic_niah_v5_indexed_counter_early_stop_patch.py \
    --input "$OUTPUT_ROOT/confirmation" \
    --phase confirmation \
    --expected-seeds "${CONFIRMATION_SEEDS[@]}" \
    --frozen-layer "$SELECTED_LAYER" \
    --output "$OUTPUT_ROOT/analysis_confirmation" 2>&1 | tee -a "$LOG"
fi

"$PYTHON" - "$MODEL" "$OUTPUT_ROOT" <<'PY'
import datetime as dt
import json
import os
import pathlib
import sys

model, root_raw = sys.argv[1:]
root = pathlib.Path(root_raw)
discovery = json.loads((root / "analysis_discovery/claim_gates.json").read_text())
confirmation = json.loads((root / "analysis_confirmation/claim_gates.json").read_text())
passed = bool(confirmation["old_html_explicit_progress_state_restoration_pass"])
value = {
    "schema_version": "realistic_niah_v5_indexed_counter_patch_complete_v2",
    "status": "PASS",
    "model_label": model,
    "discovery_seed_count": 20,
    "confirmation_seed_count": 10,
    "source_gold_count": 10,
    "prompt_modified": False,
    "patch_geometry": "full_trace_item_same_position",
    "patch_layer_mode": "single_decoder_block_input",
    "upper_layers_recomputed_after_patch": True,
    "readout_mode": "immediate_item_k_minimal_native_terminal_suffix",
    "visible_progress_confound_allowed": True,
    "internal_counter_without_visible_index_claim_allowed": False,
    "controlled_running_state_confirmation_pass": passed,
    "discovery": discovery,
    "confirmation": confirmation,
    "completed_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
}
path = root / "indexed_counter_patch_complete.json"
temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
temporary.replace(path)
PY

echo "INDEXED_COUNTER_PATCH_COMPLETE model=$MODEL utc=$(date -u +%FT%TZ)" | tee -a "$LOG"
