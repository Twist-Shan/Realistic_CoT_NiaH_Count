#!/usr/bin/env bash
set -euo pipefail

echo "DISABLED_PROMPT_MODIFICATION: this historical supervisor changes the frozen prompt" >&2
exit 64

ROOT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$ROOT_DIR"
PYTHON="$ROOT_DIR/.venv/bin/python"
GPU_INDEX=${1:-1}
POOL="$ROOT_DIR/work/v5_native_count_stream/natural_no_enumeration_seed_pool_20260823_v1"
SUPPLEMENT_GENERATION="$POOL/Gemma4-E4B_generation"
BASE_ROOT="$ROOT_DIR/work/v5_native_count_stream/natural_no_enumeration_restore_20d10c_20260823_v2/Gemma4-E4B"
OUTPUT_ROOT="$ROOT_DIR/work/v5_native_count_stream/natural_no_enumeration_restore_20d10c_20260823_v3/Gemma4-E4B"
LOG="$OUTPUT_ROOT/logs/supervisor.log"
LOCK="$OUTPUT_ROOT/locks/supervisor.lock"
mkdir -p "$OUTPUT_ROOT/logs" "$OUTPUT_ROOT/locks"

exec 9>"$LOCK"
flock -n 9 || { echo "another Gemma supplement supervisor owns the lock" >&2; exit 3; }

if ! test -s "$SUPPLEMENT_GENERATION/manifest.json"; then
  CUDA_VISIBLE_DEVICES="$GPU_INDEX" "$PYTHON" \
    scripts/run_realistic_niah_v5_natural_unnumbered_generation.py \
    --model Gemma4-E4B \
    --cache-dir "$ROOT_DIR/work/hf_cache" \
    --device-map auto \
    --torch-dtype bfloat16 \
    --attention-backend sdpa \
    --source-stimuli "$POOL/stimuli.jsonl" \
    --source-counts 10 9 \
    --planned-seeds 1264 1265 1266 1267 1268 1269 1270 1271 1272 1273 \
                    1274 1275 1276 1277 1278 1279 1280 1281 1282 1283 1284 1285 1286 1287 \
    --attempt-set reasoning_bullet_prefix \
    --resume \
    --output "$SUPPLEMENT_GENERATION" 2>&1 | tee -a "$LOG"
fi

if ! test -s "$OUTPUT_ROOT/cohort/manifest.json"; then
  "$PYTHON" scripts/assemble_realistic_niah_v5_natural_format_cohort.py \
    --base-selected "$BASE_ROOT/generation/selected_generations.jsonl" \
    --supplement-selected "$SUPPLEMENT_GENERATION/selected_generations.jsonl" \
    --output "$OUTPUT_ROOT/cohort" 2>&1 | tee -a "$LOG"
fi

mapfile -t DISCOVERY_SEEDS < <("$PYTHON" -c \
  'import json,sys; print(*json.load(open(sys.argv[1]))["discovery_seeds"], sep="\n")' \
  "$OUTPUT_ROOT/cohort/manifest.json")
mapfile -t CONFIRMATION_SEEDS < <("$PYTHON" -c \
  'import json,sys; print(*json.load(open(sys.argv[1]))["confirmation_seeds"], sep="\n")' \
  "$OUTPUT_ROOT/cohort/manifest.json")
test "${#DISCOVERY_SEEDS[@]}" -eq 20
test "${#CONFIRMATION_SEEDS[@]}" -eq 10

if ! test -s "$OUTPUT_ROOT/discovery/manifest.json"; then
  CUDA_VISIBLE_DEVICES="$GPU_INDEX" "$PYTHON" \
    scripts/run_realistic_niah_v5_unnumbered_counter_restore.py \
    --model Gemma4-E4B \
    --cache-dir "$ROOT_DIR/work/hf_cache" \
    --device-map auto \
    --torch-dtype bfloat16 \
    --attention-backend sdpa \
    --generations "$OUTPUT_ROOT/cohort/selected_generations.jsonl" \
    --phase discovery \
    --expected-seeds "${DISCOVERY_SEEDS[@]}" \
    --source-layers 16 20 24 28 32 36 \
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
    scripts/run_realistic_niah_v5_unnumbered_counter_restore.py \
    --model Gemma4-E4B \
    --cache-dir "$ROOT_DIR/work/hf_cache" \
    --device-map auto \
    --torch-dtype bfloat16 \
    --attention-backend sdpa \
    --generations "$OUTPUT_ROOT/cohort/selected_generations.jsonl" \
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

"$PYTHON" - "$OUTPUT_ROOT" <<'PY'
import datetime as dt
import json
import os
import pathlib
import sys

root = pathlib.Path(sys.argv[1])
cohort = json.loads((root / "cohort/manifest.json").read_text())
discovery = json.loads((root / "analysis_discovery/claim_gates.json").read_text())
confirmation = json.loads((root / "analysis_confirmation/claim_gates.json").read_text())
value = {
    "schema_version": "realistic_niah_v5_natural_no_enumeration_complete_v2",
    "status": "PASS",
    "model_label": "Gemma4-E4B",
    "discovery_seed_count": 20,
    "confirmation_seed_count": 10,
    "trace_tokens_model_generated": True,
    "teacher_forcing": False,
    "fixed_prefix_contains_count_information": False,
    "format_only_independent_seed_supplement": True,
    "outcome_blind": True,
    "selection_rank_used": False,
    "cohort": cohort,
    "discovery": discovery,
    "confirmation": confirmation,
    "completed_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
}
path = root / "natural_no_enumeration_complete.json"
temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
temporary.replace(path)
PY
