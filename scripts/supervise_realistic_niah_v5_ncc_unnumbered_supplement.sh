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
    ANCHOR="$ROOT_DIR/work/v5_supplement_inputs/qwen_grammar_span_anchor_panel.jsonl"
    TARGETED="$ROOT_DIR/work/v5_supplement_inputs/qwen_targeted_final_transition_registry.jsonl"
    BANK_PLAN="$ROOT_DIR/work/v5_supplement_inputs/qwen_frozen_targeted_count_plan.csv"
    GENERATIONS="$ROOT_DIR/work/v5_supplement_inputs/Qwen3-8B_generations_reparsed.jsonl"
    DISCOVERY_LAYERS=(18 22 26 30)
    ;;
  Gemma4-E4B)
    SOURCE_LAYER=16
    BANK_SIZE=6
    ANCHOR="$ROOT_DIR/work/v5_supplement_inputs/gemma_grammar_span_anchor_panel.jsonl"
    TARGETED="$ROOT_DIR/work/v5_supplement_inputs/gemma_targeted_final_transition_registry.jsonl"
    BANK_PLAN="$ROOT_DIR/work/v5_supplement_inputs/gemma_top6_frozen_targeted_count_plan.csv"
    GENERATIONS="$ROOT_DIR/work/v5_supplement_inputs/Gemma4-E4B_generations_reparsed.jsonl"
    DISCOVERY_LAYERS=(16 20 24 28 32 36)
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
OUTPUT_ROOT="$ROOT_DIR/work/v5_native_count_stream/ncc_unnumbered_supplement_20d10c_20260823_v1/$MODEL"
LOG="$OUTPUT_ROOT/logs/supervisor.log"
LOCK="$OUTPUT_ROOT/locks/supervisor.lock"
COMPLETE="$OUTPUT_ROOT/supplement_complete.json"
mkdir -p "$OUTPUT_ROOT/logs" "$OUTPUT_ROOT/locks"

exec 9>"$LOCK"
if ! flock -n 9; then
  echo "another $MODEL supplement supervisor owns the lock" >&2
  exit 3
fi

for path in "$ANCHOR" "$TARGETED" "$BANK_PLAN" "$GENERATIONS"; do
  test -s "$path" || { echo "missing input: $path" >&2; exit 4; }
done

run_ncc_phase() {
  local role=$1
  local phase=$2
  local mechanism=$3
  local output="$OUTPUT_ROOT/ncc/$phase"
  echo "NCC_START model=$MODEL phase=$phase utc=$(date -u +%FT%TZ)" | tee -a "$LOG"
  CUDA_VISIBLE_DEVICES="$GPU_INDEX" "$PYTHON" \
    "$ROOT_DIR/scripts/run_realistic_niah_v5_targeted_counter_ncc.py" \
    --mechanism-config "$mechanism" \
    --v5-config "$V5_CONFIG" \
    --model "$MODEL" \
    --cache-dir "$ROOT_DIR/work/hf_cache" \
    --device-map auto \
    --torch-dtype bfloat16 \
    --attention-backend sdpa \
    --generations "$GENERATIONS" \
    --seed-role "$role" \
    --anchor-registry "$ANCHOR" \
    --targeted-registry "$TARGETED" \
    --bank-plan "$BANK_PLAN" \
    --source-layer "$SOURCE_LAYER" \
    --resume \
    --output "$output" 2>&1 | tee -a "$LOG"
  echo "NCC_SEALED model=$MODEL phase=$phase utc=$(date -u +%FT%TZ)" | tee -a "$LOG"
}

run_ncc_phase development discovery "$MECH_DEV"
run_ncc_phase confirmation confirmation "$MECH_CONFIRM"
"$PYTHON" "$ROOT_DIR/scripts/analyze_realistic_niah_v5_targeted_counter_ncc.py" \
  --discovery "$OUTPUT_ROOT/ncc/discovery" \
  --confirmation "$OUTPUT_ROOT/ncc/confirmation" \
  --output "$OUTPUT_ROOT/ncc/analysis" 2>&1 | tee -a "$LOG"

echo "UNNUMBERED_GENERATION_START model=$MODEL utc=$(date -u +%FT%TZ)" | tee -a "$LOG"
"$PYTHON" \
  "$ROOT_DIR/scripts/run_realistic_niah_v5_unnumbered_generation.py" \
  --model "$MODEL" \
  --cache-dir "$ROOT_DIR/work/hf_cache" \
  --source-generations "$GENERATIONS" \
  --resume \
  --output "$OUTPUT_ROOT/unnumbered_counterfactual_v2/generation" 2>&1 | tee -a "$LOG"

echo "UNNUMBERED_DISCOVERY_START model=$MODEL utc=$(date -u +%FT%TZ)" | tee -a "$LOG"
CUDA_VISIBLE_DEVICES="$GPU_INDEX" "$PYTHON" \
  "$ROOT_DIR/scripts/run_realistic_niah_v5_unnumbered_counter_restore.py" \
  --model "$MODEL" \
  --cache-dir "$ROOT_DIR/work/hf_cache" \
  --device-map auto \
  --torch-dtype bfloat16 \
  --attention-backend sdpa \
  --generations "$OUTPUT_ROOT/unnumbered_counterfactual_v2/generation/selected_generations.jsonl" \
  --phase discovery \
  --source-layers "${DISCOVERY_LAYERS[@]}" \
  --resume \
  --output "$OUTPUT_ROOT/unnumbered_counterfactual_v2/discovery" 2>&1 | tee -a "$LOG"
"$PYTHON" "$ROOT_DIR/scripts/analyze_realistic_niah_v5_unnumbered_counter_restore.py" \
  --input "$OUTPUT_ROOT/unnumbered_counterfactual_v2/discovery" \
  --phase discovery \
  --output "$OUTPUT_ROOT/unnumbered_counterfactual_v2/analysis_discovery" 2>&1 | tee -a "$LOG"

SELECTED_LAYER=$("$PYTHON" -c 'import json,sys; print(json.load(open(sys.argv[1]))["selected_layer"])' \
  "$OUTPUT_ROOT/unnumbered_counterfactual_v2/analysis_discovery/claim_gates.json")
echo "UNNUMBERED_CONFIRMATION_START model=$MODEL layer=$SELECTED_LAYER utc=$(date -u +%FT%TZ)" | tee -a "$LOG"
CUDA_VISIBLE_DEVICES="$GPU_INDEX" "$PYTHON" \
  "$ROOT_DIR/scripts/run_realistic_niah_v5_unnumbered_counter_restore.py" \
  --model "$MODEL" \
  --cache-dir "$ROOT_DIR/work/hf_cache" \
  --device-map auto \
  --torch-dtype bfloat16 \
  --attention-backend sdpa \
  --generations "$OUTPUT_ROOT/unnumbered_counterfactual_v2/generation/selected_generations.jsonl" \
  --phase confirmation \
  --source-layers "$SELECTED_LAYER" \
  --resume \
  --output "$OUTPUT_ROOT/unnumbered_counterfactual_v2/confirmation" 2>&1 | tee -a "$LOG"
"$PYTHON" "$ROOT_DIR/scripts/analyze_realistic_niah_v5_unnumbered_counter_restore.py" \
  --input "$OUTPUT_ROOT/unnumbered_counterfactual_v2/confirmation" \
  --phase confirmation \
  --frozen-layer "$SELECTED_LAYER" \
  --output "$OUTPUT_ROOT/unnumbered_counterfactual_v2/analysis_confirmation" 2>&1 | tee -a "$LOG"

"$PYTHON" - "$MODEL" "$BANK_SIZE" "$OUTPUT_ROOT" "$COMPLETE" <<'PY'
import datetime as dt
import json
import os
import pathlib
import sys

model, bank_size, root_raw, complete_raw = sys.argv[1:]
root = pathlib.Path(root_raw)
ncc = json.loads((root / "ncc/analysis/claim_gates.json").read_text())
discovery = json.loads((root / "unnumbered_counterfactual_v2/analysis_discovery/claim_gates.json").read_text())
confirmation = json.loads((root / "unnumbered_counterfactual_v2/analysis_confirmation/claim_gates.json").read_text())
value = {
    "schema_version": "realistic_niah_v5_ncc_unnumbered_supplement_complete_v1",
    "status": "PASS",
    "model_label": model,
    "targeted_bank_size": int(bank_size),
    "discovery_seed_count": 20,
    "confirmation_seed_count": 10,
    "outcome_blind": True,
    "selection_rank_used": False,
    "unnumbered_panel_kind": "teacher_forced_unnumbered_gold_bullets",
    "natural_generation_claim_allowed": False,
    "controlled_hidden_state_sufficiency_claim_allowed": True,
    "ncc": ncc,
    "unnumbered_discovery": discovery,
    "unnumbered_confirmation": confirmation,
    "confirmed_internal_counter_magnitude_pass": bool(
        confirmation["old_html_internal_counter_magnitude_pass"]
    ),
    "completed_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
}
path = pathlib.Path(complete_raw)
temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
temporary.replace(path)
print(json.dumps({"status": "PASS", "model": model}, sort_keys=True))
PY
echo "SUPPLEMENT_COMPLETE model=$MODEL utc=$(date -u +%FT%TZ)" | tee -a "$LOG"
