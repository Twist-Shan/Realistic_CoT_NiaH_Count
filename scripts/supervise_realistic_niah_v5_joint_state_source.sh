#!/usr/bin/env bash
set -euo pipefail

MODEL="${1:?usage: $0 Qwen3-8B|Gemma4-E4B}"
case "$MODEL" in
  Qwen3-8B) SOURCE_LAYER=19 ;;
  Gemma4-E4B) SOURCE_LAYER=16 ;;
  *) echo "unsupported model: $MODEL" >&2; exit 2 ;;
esac

CODE_ROOT="${CODE_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
PYTHON="${PYTHON:-$CODE_ROOT/.venv/bin/python}"
CACHE_DIR="${CACHE_DIR:-$CODE_ROOT/work/hf_cache}"
RUN_ROOT="${RUN_ROOT:-$CODE_ROOT/work/v5_native_count_stream/joint_20d10c_20260820/$MODEL}"
RUNNER="$CODE_ROOT/scripts/run_realistic_niah_v5_count_stream.py"
ANALYZER="$CODE_ROOT/scripts/analyze_realistic_niah_v5_joint_state_source.py"
PATCH_ANALYZER="$CODE_ROOT/scripts/analyze_realistic_niah_v5_full_state_patch_source.py"
MECHANISM="$CODE_ROOT/configs/realistic_niah_v5_native_count_stream_dev.json"
V5_CONFIG="$CODE_ROOT/configs/realistic_niah_v5.json"
GENERATIONS="$CODE_ROOT/work/v5_trace_parser_v2/${MODEL}_generations_reparsed.jsonl"
BASIS="$CODE_ROOT/work/v5_native_count_stream/stage1_20d10c_20260820/$MODEL/running_basis.npz"
TERMINAL_PLAN_ROOT="$RUN_ROOT/terminal_last_plan"
TERMINAL_PLAN="$TERMINAL_PLAN_ROOT/terminal_last_pair_plan.csv"
PATCH_TRIALS="$RUN_ROOT/full_state_suffix8_x_answer_source"
PATCH_ANALYSIS="$RUN_ROOT/full_state_suffix8_analysis"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export HF_HOME="$CACHE_DIR"
export TOKENIZERS_PARALLELISM=false

mkdir -p "$RUN_ROOT/logs" "$RUN_ROOT/locks" "$CACHE_DIR"
LOG="$RUN_ROOT/logs/joint_state_source.log"
exec > >(tee -a "$LOG") 2>&1
exec 9>"$RUN_ROOT/locks/joint_state_source.lock"
if ! flock -n 9; then
  echo "another $MODEL joint-state-source supervisor owns the lock" >&2
  exit 75
fi

for path in "$PYTHON" "$RUNNER" "$ANALYZER" "$PATCH_ANALYZER" "$MECHANISM" \
  "$V5_CONFIG" "$GENERATIONS" "$BASIS" "${BASIS%.npz}.json"; do
  test -s "$path"
done

echo "START model=$MODEL layer=$SOURCE_LAYER gpu=$CUDA_VISIBLE_DEVICES utc=$(date -u +%FT%TZ)"
cd "$CODE_ROOT"

"$PYTHON" "$RUNNER" plan-terminal-last-patch \
  --mechanism-config "$MECHANISM" \
  --v5-config "$V5_CONFIG" \
  --model "$MODEL" \
  --generations "$GENERATIONS" \
  --cohort one_to_one \
  --seeds-per-cell 10 \
  --output "$TERMINAL_PLAN_ROOT"

# Primary parallel/serial test: five outcome-blind seeds in every agreed
# count x {-1,-3,-5} cell, terminal suffix-8 full-state clamp, followed by
# clean/prompt/trace answer-query source branches.
"$PYTHON" "$RUNNER" full-state-patch-source \
  --mechanism-config "$MECHANISM" \
  --v5-config "$V5_CONFIG" \
  --model "$MODEL" \
  --cache-dir "$CACHE_DIR" \
  --device-map auto \
  --torch-dtype bfloat16 \
  --attention-backend sdpa \
  --generations "$GENERATIONS" \
  --seed-role development \
  --cohort one_to_one \
  --row-panel trace_patch \
  --pair-plan "$TERMINAL_PLAN" \
  --layer "$SOURCE_LAYER" \
  --geometries suffix8 \
  --layer-mode cumulative_clamp \
  --patch-conditions self_patch full_donor_patch \
  --mask-conditions clean block_trace_items \
    block_trace_items_matched_control block_prompt_records \
    block_prompt_records_matched_control \
  --mask-application answer_query_only \
  --max-selection-rank 5 \
  --max-new-tokens 16 \
  --output "$PATCH_TRIALS"

"$PYTHON" "$PATCH_ANALYZER" \
  --trials "$PATCH_TRIALS" \
  --output "$PATCH_ANALYSIS" \
  --bootstrap-samples 10000 \
  --random-seed 20260820

run_scope() {
  local scope="$1"
  local trials="$RUN_ROOT/${scope}_item_end_state_x_answer_source"
  local analysis="$RUN_ROOT/${scope}_analysis"
  "$PYTHON" "$RUNNER" joint-state-source \
    --mechanism-config "$MECHANISM" \
    --v5-config "$V5_CONFIG" \
    --model "$MODEL" \
    --cache-dir "$CACHE_DIR" \
    --device-map auto \
    --torch-dtype bfloat16 \
    --attention-backend sdpa \
    --generations "$GENERATIONS" \
    --seed-role development \
    --cohort one_to_one \
    --row-panel broad_ranking \
    --basis "$BASIS" \
    --source-layer "$SOURCE_LAYER" \
    --state-scope "$scope" \
    --state-conditions clean aligned_running_state_removal \
      norm_matched_orthogonal_removal \
    --mask-conditions clean block_trace_items \
      block_trace_items_matched_control block_prompt_records \
      block_prompt_records_matched_control \
    --mask-application answer_query_only \
    --max-new-tokens 16 \
    --output "$trials"

  "$PYTHON" "$ANALYZER" \
    --trials "$trials" \
    --output "$analysis" \
    --bootstrap-samples 10000 \
    --random-seed 20260820
}

# Auxiliary specificity test: remove the discovery-fitted progress/count
# subspace at every item endpoint and compare a norm-matched orthogonal arm.
run_scope all

"$PYTHON" - "$MODEL" "$SOURCE_LAYER" "$RUN_ROOT" "$BASIS" <<'PY'
import datetime as dt
import hashlib
import json
import pathlib
import sys

model, layer, root_raw, basis_raw = sys.argv[1:]
root = pathlib.Path(root_raw)
basis = pathlib.Path(basis_raw)
patch_audit = json.loads(
    (root / "full_state_suffix8_analysis" / "audit.json").read_text(encoding="utf-8")
)
assert patch_audit["status"] == "PASS", patch_audit
state_audit = json.loads(
    (root / "all_analysis" / "audit.json").read_text(encoding="utf-8")
)
assert state_audit["status"] == "PASS", state_audit
summary = {
    "schema_version": "realistic_niah_v5_joint_state_source_supervisor_v1",
    "status": "PASS",
    "model_label": model,
    "source_layer": int(layer),
    "basis": str(basis.resolve()),
    "basis_sha256": hashlib.sha256(basis.read_bytes()).hexdigest(),
    "completed_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
    "primary_full_state_patch_audit": patch_audit,
    "auxiliary_all_endpoint_state_audit": state_audit,
}
temporary = root / ".joint_complete.json.tmp"
temporary.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
temporary.replace(root / "joint_complete.json")
print(json.dumps(summary, sort_keys=True))
PY

echo "PASS model=$MODEL utc=$(date -u +%FT%TZ)"
