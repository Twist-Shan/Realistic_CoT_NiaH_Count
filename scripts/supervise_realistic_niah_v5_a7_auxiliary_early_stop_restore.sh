#!/usr/bin/env bash
set -euo pipefail

MODEL=${1:?usage: $0 MODEL GPU_INDEX}
GPU_INDEX=${2:?usage: $0 MODEL GPU_INDEX}
ROOT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$ROOT_DIR"
PYTHON="$ROOT_DIR/.venv/bin/python"
POOL="$ROOT_DIR/work/v5_native_count_stream/a7_auxiliary_seed_pool_20260823_v1"
OUTPUT_VERSION=a7_auxiliary_early_stop_restore_20d10c_20260823_v1

case "$MODEL" in
  Qwen3-8B)
    BASE_SELECTED="$ROOT_DIR/work/v5_native_count_stream/natural_no_enumeration_restore_20d10c_20260823_v1/Qwen3-8B/generation/selected_generations.jsonl"
    DISCOVERY_LAYERS=(18 22 26 30)
    ;;
  Gemma4-E4B)
    BASE_SELECTED="$ROOT_DIR/work/v5_native_count_stream/natural_no_enumeration_restore_20d10c_20260823_v3/Gemma4-E4B/cohort/selected_generations.jsonl"
    DISCOVERY_LAYERS=(16 20 24 28 32 36)
    ;;
  *)
    echo "unsupported model: $MODEL" >&2
    exit 2
    ;;
esac

OUTPUT_ROOT="$ROOT_DIR/work/v5_native_count_stream/$OUTPUT_VERSION/$MODEL"
GENERATION="$OUTPUT_ROOT/generation_supplement"
COHORT="$OUTPUT_ROOT/cohort"
LOG="$OUTPUT_ROOT/logs/supervisor.log"
LOCK="$OUTPUT_ROOT/locks/supervisor.lock"
COMPLETE="$OUTPUT_ROOT/a7_auxiliary_complete.json"
mkdir -p "$OUTPUT_ROOT/logs" "$OUTPUT_ROOT/locks"
exec 9>"$LOCK"
flock -n 9 || { echo "another $MODEL A7 auxiliary supervisor owns the lock" >&2; exit 3; }
test -s "$POOL/stimuli.jsonl"
test -s "$BASE_SELECTED"
if test -s "$COMPLETE"; then
  echo "already complete: $COMPLETE"
  exit 0
fi

SUPPLEMENT_DISCOVERY=(1288 1289 1290 1291 1292 1293 1294 1295 1296 1297 1298 1299 1300 1301 1302 1303 1304 1305 1306 1307 1308 1309 1310 1311 1312 1313 1314 1315 1316 1317)
SUPPLEMENT_CONFIRMATION=(1318 1319 1320 1321 1322 1323 1324 1325 1326 1327 1328 1329 1330 1331 1332 1333 1334 1335 1336 1337 1338 1339 1340 1341 1342 1343 1344 1345 1346 1347)

if ! test -s "$GENERATION/manifest.json"; then
  CUDA_VISIBLE_DEVICES="$GPU_INDEX" "$PYTHON" \
    scripts/run_realistic_niah_v5_natural_unnumbered_generation.py \
    --model "$MODEL" \
    --cache-dir "$ROOT_DIR/work/hf_cache" \
    --device-map auto \
    --torch-dtype bfloat16 \
    --attention-backend sdpa \
    --source-stimuli "$POOL/stimuli.jsonl" \
    --source-counts 10 9 \
    --planned-seeds "${SUPPLEMENT_DISCOVERY[@]}" "${SUPPLEMENT_CONFIRMATION[@]}" \
    --attempt-set a7_only \
    --allow-prompt-conditioned-a7-auxiliary \
    --resume \
    --output "$GENERATION" 2>&1 | tee -a "$LOG"
fi

if ! test -s "$COHORT/manifest.json"; then
  "$PYTHON" scripts/assemble_realistic_niah_v5_natural_format_cohort.py \
    --base-selected "$BASE_SELECTED" \
    --supplement-selected "$GENERATION/selected_generations.jsonl" \
    --required-attempt 7 \
    --output "$COHORT" 2>&1 | tee -a "$LOG"
fi

mapfile -t DISCOVERY_SEEDS < <("$PYTHON" -c \
  'import json,sys; print(*json.load(open(sys.argv[1]))["discovery_seeds"], sep="\n")' \
  "$COHORT/manifest.json")
mapfile -t CONFIRMATION_SEEDS < <("$PYTHON" -c \
  'import json,sys; print(*json.load(open(sys.argv[1]))["confirmation_seeds"], sep="\n")' \
  "$COHORT/manifest.json")
test "${#DISCOVERY_SEEDS[@]}" -eq 20
test "${#CONFIRMATION_SEEDS[@]}" -eq 10

if ! test -s "$OUTPUT_ROOT/discovery/manifest.json"; then
  CUDA_VISIBLE_DEVICES="$GPU_INDEX" "$PYTHON" \
    scripts/run_realistic_niah_v5_unnumbered_counter_early_stop_restore.py \
    --model "$MODEL" \
    --cache-dir "$ROOT_DIR/work/hf_cache" \
    --device-map auto \
    --torch-dtype bfloat16 \
    --attention-backend sdpa \
    --generations "$COHORT/selected_generations.jsonl" \
    --phase discovery \
    --expected-seeds "${DISCOVERY_SEEDS[@]}" \
    --source-layers "${DISCOVERY_LAYERS[@]}" \
    --allow-prompt-conditioned-a7-auxiliary \
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
    --generations "$COHORT/selected_generations.jsonl" \
    --phase confirmation \
    --expected-seeds "${CONFIRMATION_SEEDS[@]}" \
    --source-layers "$SELECTED_LAYER" \
    --allow-prompt-conditioned-a7-auxiliary \
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
value = {
    "schema_version": "realistic_niah_v5_a7_auxiliary_early_stop_complete_v1",
    "status": "PASS",
    "model_label": model,
    "discovery_seed_count": 20,
    "confirmation_seed_count": 10,
    "prompt_conditioned_a7_auxiliary": True,
    "formal_frozen_prompt_claim_allowed": False,
    "trace_tokens_model_generated": True,
    "teacher_forcing": False,
    "future_trace_items_removed": True,
    "readout_mode": "immediate_item_k_early_stop_minimal_terminal_suffix",
    "patch_layer_mode": "cumulative_clamp_source_through_last",
    "outcome_blind": True,
    "selection_rank_used": False,
    "cohort": json.loads((root / "cohort/manifest.json").read_text()),
    "discovery": json.loads((root / "analysis_discovery/claim_gates.json").read_text()),
    "confirmation": json.loads((root / "analysis_confirmation/claim_gates.json").read_text()),
    "completed_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
}
path = root / "a7_auxiliary_complete.json"
temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
temporary.replace(path)
PY
