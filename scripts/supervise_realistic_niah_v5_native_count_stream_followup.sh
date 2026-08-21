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
PYTHON="${PYTHON:-$CODE_ROOT/.venv/bin/python}"
CACHE_DIR="${CACHE_DIR:-$CODE_ROOT/work/hf_cache}"
RUN_ROOT="${RUN_ROOT:-$CODE_ROOT/work/v5_native_count_stream/followup_20260820/$MODEL}"
STAGE1_ROOT="$CODE_ROOT/work/v5_native_count_stream/stage1_20260820/$MODEL"
RUNNER="$CODE_ROOT/scripts/run_realistic_niah_v5_count_stream.py"
MECHANISM="$CODE_ROOT/configs/realistic_niah_v5_native_count_stream_dev.json"
V5_CONFIG="$CODE_ROOT/configs/realistic_niah_v5_native_count_stream_registry_v1.json"
GENERATIONS="$CODE_ROOT/work/v5_trace_parser_v2/${MODEL}_generations_reparsed.jsonl"
SPARSE_PLAN="$STAGE1_ROOT/trace_pair_plan/trace_patch_pair_plan.csv"
TERMINAL_PLAN_DIR="$RUN_ROOT/terminal_last_plan"
TERMINAL_PLAN="$TERMINAL_PLAN_DIR/terminal_last_pair_plan.csv"
BASIS="$STAGE1_ROOT/running_basis.npz"
MIDDLE_OUTPUT="$RUN_ROOT/middle_full_state_clamp_rank5"
TERMINAL_OUTPUT="$RUN_ROOT/terminal_last_full_state_clamp_rank10"
SOURCE_MASK_OUTPUT="$RUN_ROOT/source_mask_query_only_all_heads"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export HF_HOME="$CACHE_DIR"
export TOKENIZERS_PARALLELISM=false

mkdir -p "$RUN_ROOT/logs" "$RUN_ROOT/locks" "$CACHE_DIR"
LOG="$RUN_ROOT/logs/native_count_stream_followup.log"
exec > >(tee -a "$LOG") 2>&1
exec 9>"$RUN_ROOT/locks/native_count_stream_followup.lock"
if ! flock -n 9; then
  echo "another $MODEL follow-up supervisor owns the lock" >&2
  exit 75
fi

for path in "$PYTHON" "$RUNNER" "$MECHANISM" "$V5_CONFIG" \
  "$GENERATIONS" "$SPARSE_PLAN" "$BASIS" "$STAGE1_ROOT/stage1_complete.json"; do
  test -s "$path"
done

"$PYTHON" - "$STAGE1_ROOT/stage1_complete.json" <<'PY'
import json
import pathlib
import sys

value = json.loads(pathlib.Path(sys.argv[1]).read_text(encoding="utf-8"))
assert value["status"] == "PASS", value
PY

echo "START model=$MODEL gpu=$CUDA_VISIBLE_DEVICES utc=$(date -u +%FT%TZ)"
cd "$CODE_ROOT"

"$PYTHON" "$RUNNER" plan-terminal-last-patch \
  --mechanism-config "$MECHANISM" \
  --v5-config "$V5_CONFIG" \
  --model "$MODEL" \
  --generations "$GENERATIONS" \
  --cohort one_to_one \
  --seeds-per-cell 10 \
  --output "$TERMINAL_PLAN_DIR"

common_full_state_args=(
  --mechanism-config "$MECHANISM"
  --v5-config "$V5_CONFIG"
  --model "$MODEL"
  --cache-dir "$CACHE_DIR"
  --device-map auto
  --torch-dtype bfloat16
  --attention-backend sdpa
  --generations "$GENERATIONS"
  --seed-role development
  --cohort one_to_one
  --basis "$BASIS"
  --layer "$SOURCE_LAYER"
  --readout-layers "$READOUT_LAYER"
  --geometries endpoint suffix4 suffix8 full_span
  --layer-modes cumulative_clamp
  --conditions clean self_patch full_donor_patch
)

"$PYTHON" "$RUNNER" trace-full-state-patch \
  "${common_full_state_args[@]}" \
  --pair-plan "$SPARSE_PLAN" \
  --plan-kind sparse_local \
  --max-selection-rank 5 \
  --skip-greedy \
  --output "$MIDDLE_OUTPUT"

"$PYTHON" "$RUNNER" trace-full-state-patch \
  "${common_full_state_args[@]}" \
  --pair-plan "$TERMINAL_PLAN" \
  --plan-kind terminal_last \
  --max-selection-rank 10 \
  --max-new-tokens 16 \
  --output "$TERMINAL_OUTPUT"

"$PYTHON" "$RUNNER" source-mask \
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
  --conditions clean block_trace_items block_trace_items_matched_control \
    block_prompt_records block_prompt_records_matched_control \
  --mask-application answer_query_only \
  --skip-greedy \
  --output "$SOURCE_MASK_OUTPUT"

"$PYTHON" - "$MODEL" "$RUN_ROOT" <<'PY'
import datetime as dt
import json
import math
import pathlib
import sys

model, root_raw = sys.argv[1:]
root = pathlib.Path(root_raw)


def read_shards(name: str, expected_shards: int, expected_rows_per_shard: int):
    paths = sorted((root / name / "shards").glob("*.jsonl"))
    assert len(paths) == expected_shards, (name, len(paths), expected_shards)
    rows = []
    for path in paths:
        local = [
            json.loads(line)
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        assert len(local) == expected_rows_per_shard, (path, len(local))
        rows.extend(local)
    return rows


middle = read_shards("middle_full_state_clamp_rank5", 360, 3)
terminal = read_shards("terminal_last_full_state_clamp_rank10", 760, 3)
source = read_shards("source_mask_query_only_all_heads", 100, 5)

for panel, rows in (("middle", middle), ("terminal", terminal)):
    statuses = {row["status"] for row in rows}
    assert statuses <= {"ok", "not_applicable"}, (panel, statuses)
    for row in rows:
        if row["status"] == "not_applicable":
            assert row["patch_geometry"] in {"suffix4", "suffix8", "full_span"}
            continue
        assert row["patch_layer_mode"] == "cumulative_clamp"
        assert row["condition"] in {"clean", "self_patch", "full_donor_patch"}
        assert math.isfinite(float(row["expected_count"]))
        assert math.isfinite(float(row["correct_count_margin"]))
        applications = row["patch_hook_applications"]
        if row["condition"] == "clean":
            assert applications == {}
        else:
            assert applications and set(applications.values()) == {1}
    if panel == "terminal":
        assert all(
            row.get("receiver_is_terminal", True)
            for row in rows
            if row["status"] == "ok"
        )
        assert all(
            not row.get("later_trace_self_correction_possible", False)
            for row in rows
            if row["status"] == "ok"
        )

conditions = {
    "clean",
    "block_trace_items",
    "block_trace_items_matched_control",
    "block_prompt_records",
    "block_prompt_records_matched_control",
}
assert {row["condition"] for row in source} == conditions
assert {row["status"] for row in source} == {"ok"}
assert {row["mask_scope"] for row in source} == {"answer_query_only"}
assert {row["source_edge_mask_head_scope"] for row in source} == {
    "all_attention_heads"
}
for row in source:
    assert math.isfinite(float(row["expected_count"]))
    assert math.isfinite(float(row["correct_count_margin"]))

terminal_plan = root / "terminal_last_plan" / "terminal_last_pair_plan.csv"
summary = {
    "schema_version": "realistic_niah_v5_native_count_stream_followup_v1",
    "status": "PASS",
    "model_label": model,
    "completed_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
    "middle_pair_count": 90,
    "middle_geometry_shards": 360,
    "middle_condition_rows": len(middle),
    "middle_not_applicable_rows": sum(
        row["status"] == "not_applicable" for row in middle
    ),
    "terminal_last_pair_count": 190,
    "terminal_last_geometry_shards": 760,
    "terminal_last_condition_rows": len(terminal),
    "terminal_not_applicable_rows": sum(
        row["status"] == "not_applicable" for row in terminal
    ),
    "source_mask_request_count": 100,
    "source_mask_condition_rows": len(source),
    "terminal_plan": str(terminal_plan.resolve()),
}
temporary = root / ".followup_complete.json.tmp"
temporary.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
temporary.replace(root / "followup_complete.json")
print(json.dumps(summary, sort_keys=True))
PY

echo "PASS model=$MODEL utc=$(date -u +%FT%TZ)"
