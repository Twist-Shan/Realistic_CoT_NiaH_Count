#!/usr/bin/env bash
set -euo pipefail

echo "DISABLED_PROMPT_MODIFICATION: this historical supervisor changes the frozen prompt" >&2
exit 64

MODEL=${1:?usage: $0 MODEL GPU_INDEX}
GPU_INDEX=${2:?usage: $0 MODEL GPU_INDEX}
ROOT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$ROOT_DIR"

case "$MODEL" in
  Qwen3-8B)
    GENERATIONS="$ROOT_DIR/work/v5_supplement_inputs/Qwen3-8B_generations_reparsed.jsonl"
    DISCOVERY_LAYERS=(18 22 26 30)
    ATTEMPT_SET=reasoning_bullet_prefix
    SOURCE_COUNTS=(10)
    OUTPUT_VERSION=natural_no_enumeration_restore_20d10c_20260823_v1
    ;;
  Gemma4-E4B)
    GENERATIONS="$ROOT_DIR/work/v5_supplement_inputs/Gemma4-E4B_generations_reparsed.jsonl"
    DISCOVERY_LAYERS=(16 20 24 28 32 36)
    ATTEMPT_SET=reasoning_bullet_prefix
    SOURCE_COUNTS=(10 9)
    OUTPUT_VERSION=natural_no_enumeration_restore_20d10c_20260823_v2
    ;;
  *)
    echo "unsupported model: $MODEL" >&2
    exit 2
    ;;
esac

PYTHON="$ROOT_DIR/.venv/bin/python"
OUTPUT_ROOT="$ROOT_DIR/work/v5_native_count_stream/$OUTPUT_VERSION/$MODEL"
LOG="$OUTPUT_ROOT/logs/supervisor.log"
LOCK="$OUTPUT_ROOT/locks/supervisor.lock"
COMPLETE="$OUTPUT_ROOT/natural_no_enumeration_complete.json"
mkdir -p "$OUTPUT_ROOT/logs" "$OUTPUT_ROOT/locks"

exec 9>"$LOCK"
if ! flock -n 9; then
  echo "another $MODEL natural no-enumeration supervisor owns the lock" >&2
  exit 3
fi
test -s "$GENERATIONS" || { echo "missing input: $GENERATIONS" >&2; exit 4; }
if test -s "$COMPLETE"; then
  echo "already complete: $COMPLETE"
  exit 0
fi

if ! test -s "$OUTPUT_ROOT/archive_audit/audit.json"; then
  echo "ARCHIVE_AUDIT_START model=$MODEL utc=$(date -u +%FT%TZ)" | tee -a "$LOG"
  "$PYTHON" scripts/audit_realistic_niah_v5_natural_no_enumeration.py \
    --generations "$GENERATIONS" \
    --model "$MODEL" \
    --output "$OUTPUT_ROOT/archive_audit" 2>&1 | tee -a "$LOG"
fi

if ! test -s "$OUTPUT_ROOT/generation/manifest.json"; then
  echo "NATURAL_GENERATION_START model=$MODEL attempt_set=$ATTEMPT_SET utc=$(date -u +%FT%TZ)" | tee -a "$LOG"
  CUDA_VISIBLE_DEVICES="$GPU_INDEX" "$PYTHON" \
    scripts/run_realistic_niah_v5_natural_unnumbered_generation.py \
    --model "$MODEL" \
    --cache-dir "$ROOT_DIR/work/hf_cache" \
    --device-map auto \
    --torch-dtype bfloat16 \
    --attention-backend sdpa \
    --source-generations "$GENERATIONS" \
    --source-counts "${SOURCE_COUNTS[@]}" \
    --attempt-set "$ATTEMPT_SET" \
    --resume \
    --require-complete \
    --output "$OUTPUT_ROOT/generation" 2>&1 | tee -a "$LOG"
fi

if ! test -s "$OUTPUT_ROOT/discovery/manifest.json"; then
  echo "RESTORE_DISCOVERY_START model=$MODEL utc=$(date -u +%FT%TZ)" | tee -a "$LOG"
  CUDA_VISIBLE_DEVICES="$GPU_INDEX" "$PYTHON" \
    scripts/run_realistic_niah_v5_unnumbered_counter_restore.py \
    --model "$MODEL" \
    --cache-dir "$ROOT_DIR/work/hf_cache" \
    --device-map auto \
    --torch-dtype bfloat16 \
    --attention-backend sdpa \
    --generations "$OUTPUT_ROOT/generation/selected_generations.jsonl" \
    --phase discovery \
    --source-layers "${DISCOVERY_LAYERS[@]}" \
    --resume \
    --output "$OUTPUT_ROOT/discovery" 2>&1 | tee -a "$LOG"
fi
if ! test -s "$OUTPUT_ROOT/analysis_discovery/claim_gates.json"; then
  "$PYTHON" scripts/analyze_realistic_niah_v5_unnumbered_counter_restore.py \
    --input "$OUTPUT_ROOT/discovery" \
    --phase discovery \
    --output "$OUTPUT_ROOT/analysis_discovery" 2>&1 | tee -a "$LOG"
fi

SELECTED_LAYER=$("$PYTHON" -c 'import json,sys; print(json.load(open(sys.argv[1]))["selected_layer"])' \
  "$OUTPUT_ROOT/analysis_discovery/claim_gates.json")
if ! test -s "$OUTPUT_ROOT/confirmation/manifest.json"; then
  echo "RESTORE_CONFIRMATION_START model=$MODEL layer=$SELECTED_LAYER utc=$(date -u +%FT%TZ)" | tee -a "$LOG"
  CUDA_VISIBLE_DEVICES="$GPU_INDEX" "$PYTHON" \
    scripts/run_realistic_niah_v5_unnumbered_counter_restore.py \
    --model "$MODEL" \
    --cache-dir "$ROOT_DIR/work/hf_cache" \
    --device-map auto \
    --torch-dtype bfloat16 \
    --attention-backend sdpa \
    --generations "$OUTPUT_ROOT/generation/selected_generations.jsonl" \
    --phase confirmation \
    --source-layers "$SELECTED_LAYER" \
    --resume \
    --output "$OUTPUT_ROOT/confirmation" 2>&1 | tee -a "$LOG"
fi
if ! test -s "$OUTPUT_ROOT/analysis_confirmation/claim_gates.json"; then
  "$PYTHON" scripts/analyze_realistic_niah_v5_unnumbered_counter_restore.py \
    --input "$OUTPUT_ROOT/confirmation" \
    --phase confirmation \
    --frozen-layer "$SELECTED_LAYER" \
    --output "$OUTPUT_ROOT/analysis_confirmation" 2>&1 | tee -a "$LOG"
fi

"$PYTHON" - "$MODEL" "$ATTEMPT_SET" "$OUTPUT_ROOT" "$COMPLETE" <<'PY'
import datetime as dt
import json
import os
import pathlib
import sys

model, attempt_set, root_raw, complete_raw = sys.argv[1:]
root = pathlib.Path(root_raw)
archive = json.loads((root / "archive_audit/audit.json").read_text())
generation = json.loads((root / "generation/manifest.json").read_text())
discovery = json.loads((root / "analysis_discovery/claim_gates.json").read_text())
confirmation = json.loads((root / "analysis_confirmation/claim_gates.json").read_text())
value = {
    "schema_version": "realistic_niah_v5_natural_no_enumeration_complete_v1",
    "status": "PASS",
    "model_label": model,
    "attempt_set": attempt_set,
    "discovery_seed_count": 20,
    "confirmation_seed_count": 10,
    "trace_tokens_model_generated": True,
    "teacher_forcing": False,
    "fixed_prefix_contains_count_information": False,
    "outcome_blind": True,
    "selection_rank_used": False,
    "archive_audit": archive,
    "generation": generation,
    "discovery": discovery,
    "confirmation": confirmation,
    "completed_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
}
path = pathlib.Path(complete_raw)
temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
temporary.replace(path)
PY
echo "NATURAL_NO_ENUMERATION_COMPLETE model=$MODEL utc=$(date -u +%FT%TZ)" | tee -a "$LOG"
