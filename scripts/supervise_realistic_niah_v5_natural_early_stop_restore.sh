#!/usr/bin/env bash
set -euo pipefail

echo "DISABLED_PROMPT_MODIFICATION: its input banks do not preserve the frozen prompt" >&2
exit 64

MODEL=${1:?usage: $0 MODEL GPU_INDEX}
GPU_INDEX=${2:?usage: $0 MODEL GPU_INDEX}
ROOT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$ROOT_DIR"
PYTHON="$ROOT_DIR/.venv/bin/python"
OUTPUT_VERSION=natural_no_enumeration_early_stop_restore_20d10c_20260823_v1

case "$MODEL" in
  Qwen3-8B)
    GENERATIONS="$ROOT_DIR/work/v5_native_count_stream/natural_no_enumeration_restore_20d10c_20260823_v1/Qwen3-8B/generation/selected_generations.jsonl"
    DISCOVERY_SEEDS=(1234 1235 1236 1237 1238 1239 1240 1241 1242 1243 1244 1245 1246 1247 1248 1249 1250 1251 1252 1253)
    CONFIRMATION_SEEDS=(1254 1255 1256 1257 1258 1259 1260 1261 1262 1263)
    DISCOVERY_LAYERS=(18 22 26 30)
    ;;
  Gemma4-E4B)
    GENERATIONS="$ROOT_DIR/work/v5_native_count_stream/natural_no_enumeration_restore_20d10c_20260823_v3/Gemma4-E4B/cohort/selected_generations.jsonl"
    DISCOVERY_SEEDS=(1234 1235 1236 1237 1238 1239 1240 1241 1242 1243 1244 1245 1246 1249 1251 1252 1253 1264 1265 1267)
    CONFIRMATION_SEEDS=(1254 1255 1258 1259 1260 1261 1274 1275 1276 1277)
    DISCOVERY_LAYERS=(16 20 24 28 32 36)
    ;;
  *)
    echo "unsupported model: $MODEL" >&2
    exit 2
    ;;
esac

OUTPUT_ROOT="$ROOT_DIR/work/v5_native_count_stream/$OUTPUT_VERSION/$MODEL"
LOG="$OUTPUT_ROOT/logs/supervisor.log"
LOCK="$OUTPUT_ROOT/locks/supervisor.lock"
COMPLETE="$OUTPUT_ROOT/early_stop_complete.json"
mkdir -p "$OUTPUT_ROOT/logs" "$OUTPUT_ROOT/locks"
exec 9>"$LOCK"
flock -n 9 || { echo "another $MODEL early-stop supervisor owns the lock" >&2; exit 3; }
test -s "$GENERATIONS"
if test -s "$COMPLETE"; then
  echo "already complete: $COMPLETE"
  exit 0
fi

if ! test -s "$OUTPUT_ROOT/discovery/manifest.json"; then
  CUDA_VISIBLE_DEVICES="$GPU_INDEX" "$PYTHON" \
    scripts/run_realistic_niah_v5_unnumbered_counter_early_stop_restore.py \
    --model "$MODEL" \
    --cache-dir "$ROOT_DIR/work/hf_cache" \
    --device-map auto \
    --torch-dtype bfloat16 \
    --attention-backend sdpa \
    --generations "$GENERATIONS" \
    --phase discovery \
    --expected-seeds "${DISCOVERY_SEEDS[@]}" \
    --source-layers "${DISCOVERY_LAYERS[@]}" \
    --resume \
    --output "$OUTPUT_ROOT/discovery" 2>&1 | tee -a "$LOG"
fi
if ! test -s "$OUTPUT_ROOT/analysis_discovery/claim_gates.json"; then
  "$PYTHON" scripts/analyze_realistic_niah_v5_unnumbered_counter_restore.py \
    --input "$OUTPUT_ROOT/discovery" \
    --phase discovery \
    --expected-seeds "${DISCOVERY_SEEDS[@]}" \
    --output "$OUTPUT_ROOT/analysis_discovery" 2>&1 | tee -a "$LOG"
fi

SELECTED_LAYER=$("$PYTHON" -c \
  'import json,sys; print(json.load(open(sys.argv[1]))["selected_layer"])' \
  "$OUTPUT_ROOT/analysis_discovery/claim_gates.json")
if ! test -s "$OUTPUT_ROOT/confirmation/manifest.json"; then
  CUDA_VISIBLE_DEVICES="$GPU_INDEX" "$PYTHON" \
    scripts/run_realistic_niah_v5_unnumbered_counter_early_stop_restore.py \
    --model "$MODEL" \
    --cache-dir "$ROOT_DIR/work/hf_cache" \
    --device-map auto \
    --torch-dtype bfloat16 \
    --attention-backend sdpa \
    --generations "$GENERATIONS" \
    --phase confirmation \
    --expected-seeds "${CONFIRMATION_SEEDS[@]}" \
    --source-layers "$SELECTED_LAYER" \
    --resume \
    --output "$OUTPUT_ROOT/confirmation" 2>&1 | tee -a "$LOG"
fi
if ! test -s "$OUTPUT_ROOT/analysis_confirmation/claim_gates.json"; then
  "$PYTHON" scripts/analyze_realistic_niah_v5_unnumbered_counter_restore.py \
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
value = {
    "schema_version": "realistic_niah_v5_natural_early_stop_complete_v1",
    "status": "PASS",
    "model_label": model,
    "discovery_seed_count": 20,
    "confirmation_seed_count": 10,
    "trace_tokens_model_generated": True,
    "teacher_forcing": False,
    "future_trace_items_removed": True,
    "readout_mode": "immediate_item_k_early_stop_minimal_terminal_suffix",
    "patch_layer_mode": "cumulative_clamp_source_through_last",
    "outcome_blind": True,
    "selection_rank_used": False,
    "discovery": discovery,
    "confirmation": confirmation,
    "completed_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
}
path = root / "early_stop_complete.json"
temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
temporary.replace(path)
PY
