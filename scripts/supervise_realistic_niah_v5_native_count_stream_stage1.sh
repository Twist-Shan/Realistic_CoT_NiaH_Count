#!/usr/bin/env bash
set -euo pipefail

MODEL="${1:?usage: $0 Qwen3-8B|Gemma4-E4B}"
case "$MODEL" in
  Qwen3-8B)
    SOURCE_LAYER=18
    READOUT_LAYER=19
    ;;
  Gemma4-E4B)
    SOURCE_LAYER=16
    READOUT_LAYER=17
    ;;
  *)
    echo "unsupported model: $MODEL" >&2
    exit 2
    ;;
esac

CODE_ROOT="${CODE_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
RUN_ROOT="${RUN_ROOT:-$CODE_ROOT/work/v5_native_count_stream/stage1_20260820/$MODEL}"
if [[ -x "$CODE_ROOT/.venv/bin/python" ]]; then
  DEFAULT_PYTHON="$CODE_ROOT/.venv/bin/python"
else
  DEFAULT_PYTHON="$(command -v python)"
fi
PYTHON="${PYTHON:-$DEFAULT_PYTHON}"
CACHE_DIR="${CACHE_DIR:-$CODE_ROOT/work/hf_cache}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export HF_HOME="$CACHE_DIR"
export TOKENIZERS_PARALLELISM=false

RUNNER="$CODE_ROOT/scripts/run_realistic_niah_v5_count_stream.py"
MECHANISM="$CODE_ROOT/configs/realistic_niah_v5_native_count_stream_dev.json"
V5_CONFIG="$CODE_ROOT/configs/realistic_niah_v5_native_count_stream_registry_v1.json"
GENERATIONS="$CODE_ROOT/work/v5_trace_parser_v2/${MODEL}_generations_reparsed.jsonl"
CAPTURE_INDEX="$CODE_ROOT/work/v5_geometry_full_panel/running/$MODEL/capture_index.jsonl"

mkdir -p "$RUN_ROOT/logs" "$RUN_ROOT/locks" "$CACHE_DIR"
LOG="$RUN_ROOT/logs/native_count_stream_stage1.log"
exec > >(tee -a "$LOG") 2>&1
exec 9>"$RUN_ROOT/locks/native_count_stream_stage1.lock"
if ! flock -n 9; then
  echo "another $MODEL count-stream stage-1 supervisor owns the lock" >&2
  exit 75
fi

for path in "$PYTHON" "$RUNNER" "$MECHANISM" "$V5_CONFIG" \
  "$GENERATIONS"; do
  test -e "$path"
done

echo "START model=$MODEL gpu=$CUDA_VISIBLE_DEVICES utc=$(date -u +%FT%TZ)"
cd "$CODE_ROOT"

TRACE_PLAN_DIR="$RUN_ROOT/trace_pair_plan"
TRACE_PLAN="$TRACE_PLAN_DIR/trace_patch_pair_plan.csv"
BASIS="$RUN_ROOT/running_basis.npz"
BROAD_CAPTURE="$RUN_ROOT/broad_ranking_capture"
TRACE_PLAN_BANK="$RUN_ROOT/broad_plan_trace"
PROMPT_PLAN_BANK="$RUN_ROOT/broad_plan_prompt"
TRACE_TRIALS="$RUN_ROOT/broad_k_grid_trace"
PROMPT_TRIALS="$RUN_ROOT/broad_k_grid_prompt"

"$PYTHON" "$RUNNER" plan-trace-patch \
  --mechanism-config "$MECHANISM" \
  --v5-config "$V5_CONFIG" \
  --model "$MODEL" \
  --generations "$GENERATIONS" \
  --cohort one_to_one \
  --output "$TRACE_PLAN_DIR"

if [[ -s "$BASIS" ]]; then
  "$PYTHON" - "$BASIS" "$SOURCE_LAYER" "$READOUT_LAYER" <<'PY'
import sys

import numpy as np

path, source_layer, readout_layer = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
with np.load(path, allow_pickle=False) as artifact:
    expected = {
        f"center_L{source_layer}",
        f"basis_L{source_layer}",
        f"control_basis_L{source_layer}",
        f"center_L{readout_layer}",
        f"basis_L{readout_layer}",
        f"control_basis_L{readout_layer}",
    }
    assert set(artifact.files) == expected, (artifact.files, sorted(expected))
    for layer in (source_layer, readout_layer):
        center = artifact[f"center_L{layer}"]
        basis = artifact[f"basis_L{layer}"]
        control = artifact[f"control_basis_L{layer}"]
        assert center.ndim == 1
        assert basis.shape == (center.size, 3), basis.shape
        assert control.shape == (center.size, 3), control.shape
print(f"REUSE_BASIS path={path} source_layer={source_layer} readout_layer={readout_layer}")
PY
else
  test -e "$CAPTURE_INDEX"
  "$PYTHON" "$RUNNER" fit-basis \
    --mechanism-config "$MECHANISM" \
    --capture-index "$CAPTURE_INDEX" \
    --site-kind item_end \
    --label occurrence \
    --cohort one_to_one \
    --layers "$SOURCE_LAYER" "$READOUT_LAYER" \
    --rank 3 \
    --random-seed 20260820 \
    --output "$BASIS"
fi

"$PYTHON" "$RUNNER" capture-broad \
  --mechanism-config "$MECHANISM" \
  --v5-config "$V5_CONFIG" \
  --model "$MODEL" \
  --cache-dir "$CACHE_DIR" \
  --device-map auto \
  --torch-dtype bfloat16 \
  --attention-backend sdpa \
  --generations "$GENERATIONS" \
  --seed-role development \
  --cohort parser_hit \
  --row-panel broad_ranking \
  --output "$BROAD_CAPTURE"

for SOURCE in trace_items prompt_records; do
  if [[ "$SOURCE" == "trace_items" ]]; then
    PLAN_DIR="$TRACE_PLAN_BANK"
  else
    PLAN_DIR="$PROMPT_PLAN_BANK"
  fi
  "$PYTHON" "$RUNNER" plan-broad \
    --mechanism-config "$MECHANISM" \
    --captures "$BROAD_CAPTURE" \
    --model "$MODEL" \
    --source-group "$SOURCE" \
    --random-seed 20260820 \
    --output "$PLAN_DIR"
done

run_k_grid() {
  local source="$1"
  local plan="$2"
  local output="$3"
  shift 3
  "$PYTHON" "$RUNNER" broad-heads \
    --mechanism-config "$MECHANISM" \
    --v5-config "$V5_CONFIG" \
    --model "$MODEL" \
    --cache-dir "$CACHE_DIR" \
    --device-map auto \
    --torch-dtype bfloat16 \
    --attention-backend sdpa \
    --generations "$GENERATIONS" \
    --seed-role development \
    --cohort parser_hit \
    --row-panel broad_k_selection \
    --plan "$plan" \
    --bank-sizes "$@" \
    --skip-greedy \
    --output "$output"
}

run_k_grid trace_items \
  "$TRACE_PLAN_BANK/answer_broad_head_plan.csv" "$TRACE_TRIALS" \
  1 2 4 8 16 32
run_k_grid prompt_records \
  "$PROMPT_PLAN_BANK/answer_broad_head_plan.csv" "$PROMPT_TRIALS" \
  1 2 4 8 16 32

select_k() {
  local source="$1"
  local plan="$2"
  local trials="$3"
  local output="$4"
  shift 4
  "$PYTHON" "$RUNNER" select-broad-k \
    --mechanism-config "$MECHANISM" \
    --model "$MODEL" \
    --source-group "$source" \
    --plan "$plan" \
    --trials "$trials" "$@" \
    --random-seed 20260820 \
    --output "$output"
}

decision_status() {
  "$PYTHON" - "$1" <<'PY'
import json
import pathlib
import sys
print(json.loads(pathlib.Path(sys.argv[1]).read_text(encoding="utf-8"))["status"])
PY
}

FINAL_TRACE_SELECTION="$RUN_ROOT/k_selection_trace"
FINAL_PROMPT_SELECTION="$RUN_ROOT/k_selection_prompt"
select_k trace_items "$TRACE_PLAN_BANK/answer_broad_head_plan.csv" \
  "$TRACE_TRIALS" "$FINAL_TRACE_SELECTION"
select_k prompt_records "$PROMPT_PLAN_BANK/answer_broad_head_plan.csv" \
  "$PROMPT_TRIALS" "$FINAL_PROMPT_SELECTION"

extend_boundary_if_needed() {
  local source="$1"
  local base_trials="$2"
  local base_selection="$3"
  local result_variable="$4"
  local status
  status="$(decision_status "$base_selection/k_selection_decision.json")"
  if [[ "$status" != "requires_boundary_extension" ]]; then
    printf -v "$result_variable" '%s' "$base_selection"
    return
  fi
  local extended_plan="$RUN_ROOT/broad_plan_${source}_with_K64"
  local trials_k64="$RUN_ROOT/broad_K64_${source}"
  local extended_selection="$RUN_ROOT/k_selection_${source}_with_K64"
  "$PYTHON" "$RUNNER" plan-broad \
    --mechanism-config "$MECHANISM" \
    --captures "$BROAD_CAPTURE" \
    --model "$MODEL" \
    --source-group "$source" \
    --bank-sizes 1 2 4 8 16 32 64 \
    --random-seed 20260820 \
    --output "$extended_plan"
  run_k_grid "$source" "$extended_plan/answer_broad_head_plan.csv" \
    "$trials_k64" 64
  select_k "$source" "$extended_plan/answer_broad_head_plan.csv" \
    "$base_trials" "$extended_selection" "$trials_k64"
  printf -v "$result_variable" '%s' "$extended_selection"
}

extend_boundary_if_needed trace_items \
  "$TRACE_TRIALS" "$FINAL_TRACE_SELECTION" FINAL_TRACE_SELECTION
extend_boundary_if_needed prompt_records \
  "$PROMPT_TRIALS" "$FINAL_PROMPT_SELECTION" FINAL_PROMPT_SELECTION

"$PYTHON" "$RUNNER" trace-patch \
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
  --pair-plan "$TRACE_PLAN" \
  --basis "$BASIS" \
  --layer "$SOURCE_LAYER" \
  --readout-layers "$READOUT_LAYER" \
  --conditions clean self_patch full_donor_patch \
    progress_projected_patch norm_matched_orthogonal_patch \
  --random-seed 20260820 \
  --skip-greedy \
  --output "$RUN_ROOT/trace_patch"

"$PYTHON" - \
  "$MODEL" "$RUN_ROOT" "$FINAL_TRACE_SELECTION" "$FINAL_PROMPT_SELECTION" <<'PY'
import datetime
import collections
import json
import pathlib
import sys

model = sys.argv[1]
root = pathlib.Path(sys.argv[2])
trace_selection = pathlib.Path(sys.argv[3])
prompt_selection = pathlib.Path(sys.argv[4])

def load(path):
    return json.loads(path.read_text(encoding="utf-8"))

capture = load(root / "broad_ranking_capture" / "manifest.json")
pair_plan = load(root / "trace_pair_plan" / "manifest.json")
trace_patch = load(root / "trace_patch" / "manifest.json")
trace_decision = load(trace_selection / "k_selection_decision.json")
prompt_decision = load(prompt_selection / "k_selection_decision.json")
assert int(capture["completed_shards"]) == 100, capture["completed_shards"]
assert int(pair_plan["pair_count"]) == 350, pair_plan["pair_count"]
assert int(pair_plan["local_pair_count"]) == 330
assert int(pair_plan["terminal_pair_count"]) == 20
assert int(trace_patch["completed_shards"]) == 350, trace_patch["completed_shards"]
assert int(trace_patch["planned_pair_count"]) == 350

for source in ("trace", "prompt"):
    trial_root = root / f"broad_k_grid_{source}"
    trial_manifest = load(trial_root / "manifest.json")
    shards = sorted((trial_root / "shards").glob("*.jsonl"))
    rows = [
        json.loads(line)
        for shard in shards
        for line in shard.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert int(trial_manifest["completed_shards"]) == 100
    assert len(shards) == 100
    assert len(rows) == 3000, (source, len(rows))
    assert {int(row["bank_size"]) for row in rows} == {1, 2, 4, 8, 16, 32}
    assert collections.Counter(str(row["condition"]) for row in rows) == {
        "clean": 600,
        "selected_bank": 600,
        "layer_matched_random": 1800,
    }

trace_shards = sorted((root / "trace_patch" / "shards").glob("*.jsonl"))
trace_rows = [
    json.loads(line)
    for shard in trace_shards
    for line in shard.read_text(encoding="utf-8").splitlines()
    if line.strip()
]
assert len(trace_shards) == 350
assert len(trace_rows) == 1750, len(trace_rows)
assert collections.Counter(str(row["condition"]) for row in trace_rows) == {
    "clean": 350,
    "self_patch": 350,
    "full_donor_patch": 350,
    "progress_projected_patch": 350,
    "norm_matched_orthogonal_patch": 350,
}
assert collections.Counter(str(row["panel_kind"]) for row in trace_rows) == {
    "local": 1650,
    "terminal": 100,
}
allowed = {"frozen_for_confirmation", "no_positive_discovery_bank"}
assert trace_decision["status"] in allowed, trace_decision
assert prompt_decision["status"] in allowed, prompt_decision
payload = {
    "schema_version": "realistic_niah_v5_native_count_stream_stage1_supervisor_v1",
    "status": "PASS",
    "model_label": model,
    "completed_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
    "broad_ranking_requests": 100,
    "trace_patch_pairs": 350,
    "trace_patch_condition_rows": 1750,
    "trace_source_k_decision": trace_decision,
    "prompt_source_k_decision": prompt_decision,
    "trace_selection_dir": str(trace_selection),
    "prompt_selection_dir": str(prompt_selection),
}
output = root / "stage1_complete.json"
output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
print(json.dumps(payload, sort_keys=True))
PY

echo "ALL_COMPLETE model=$MODEL utc=$(date -u +%FT%TZ)"
