#!/usr/bin/env bash
set -euo pipefail

CODE_ROOT="${CODE_ROOT:-/home/ubuntu/CoT-Native-thinking-v5/code/Realistic_CoT_NiaH_Count_native_v5_p0_20260820}"
OLD_RUN_ROOT="${OLD_RUN_ROOT:-/home/ubuntu/CoT-Native-thinking-v5/runs/v5_native_mechanism_reboot_20260818/Qwen3-8B}"
RUN_ROOT="${RUN_ROOT:-/home/ubuntu/CoT-Native-thinking-v5/runs/v5_native_read_write_stage_20260820/Qwen3-8B}"
PYTHON="${PYTHON:-/home/ubuntu/CoT-Native-thinking-v5/venv/bin/python}"
CACHE_DIR="${CACHE_DIR:-/home/ubuntu/CoT-Native-thinking-v5/cache/huggingface}"
CONFIG="$CODE_ROOT/configs/realistic_niah_v5.json"
GENERATIONS="$CODE_ROOT/work/v5_trace_parser_v2/Qwen3-8B_generations_reparsed.jsonl"
REFERENCE="$OLD_RUN_ROOT/head_behavior_attention_abs_postmarker_adjbefore_compact_persistent_k32_screen10_dev_v1"
P0_PLAN="$RUN_ROOT/causal_plan_single_token_adjbefore_compact_p0_k128_v1/retrieval_anchor_bank_plan.csv"
P2_PLAN="$RUN_ROOT/causal_plan_single_token_adjbefore_compact_p2_k128_v1/retrieval_anchor_bank_plan.csv"
CITYPRE_SOURCE="$OLD_RUN_ROOT/source_attention_citypre_adjbefore_compact_dev_v1"
CITYPRE_PLAN_DIR="$RUN_ROOT/causal_plan_single_token_adjbefore_compact_citypre_k128_v1"
CITYPRE_PLAN="$CITYPRE_PLAN_DIR/retrieval_anchor_bank_plan.csv"
LOG="$RUN_ROOT/logs/qwen_adjbefore_citypre_bank_single_token_k128_supervisor.log"
COMPLETE="$RUN_ROOT/qwen_adjbefore_citypre_bank_single_token_k128_complete.json"

mkdir -p "$RUN_ROOT/logs" "$RUN_ROOT/locks" "$CACHE_DIR"
exec 9>"$RUN_ROOT/locks/qwen_adjbefore_citypre_bank_single_token_k128.lock"
if ! flock -n 9; then
  echo "another Qwen city-pre single-token scan owns the lock" >&2
  exit 75
fi

cd "$CODE_ROOT"
test -f "$P0_PLAN"
test -f "$P2_PLAN"

echo "START citypre_plan utc=$(date -u +%FT%TZ)" | tee -a "$LOG"
"$PYTHON" scripts/run_realistic_niah_v5.py causal-plan \
  --config "$CONFIG" \
  --source-writes "$CITYPRE_SOURCE" \
  --output "$CITYPRE_PLAN_DIR" \
  --bank-size 128 \
  --anchor-role city_pre_d1 \
  --target-grammar-class adjacent_rank_before_city \
  --target-retrieval-surface-variant rank_before_city_compact \
  --selection-metric target_source_attention_mass \
  --selection-eligibility-scope local \
  --selection-aggregation seed_event_mean \
  --full-panel-plan \
  --selected-only-smoke 2>&1 | tee -a "$LOG"
echo "COMPLETE citypre_plan utc=$(date -u +%FT%TZ)" | tee -a "$LOG"

run_bank() {
  local bank="$1"
  local plan="$2"
  local conditions="$3"
  local output="$RUN_ROOT/head_behavior_single_token_adjbefore_compact_${bank}_at_citypre_k128_v1"
  echo "START ${bank}_at_citypre utc=$(date -u +%FT%TZ)" | tee -a "$LOG"
  env HF_HOME="$CACHE_DIR" TOKENIZERS_PARALLELISM=false \
    "$PYTHON" scripts/run_realistic_niah_v5.py causal-heads-behavior \
      --config "$CONFIG" \
      --model Qwen3-8B \
      --cache-dir "$CACHE_DIR" \
      --device-map auto \
      --torch-dtype bfloat16 \
      --attention-backend sdpa \
      --generations "$GENERATIONS" \
      --plan "$plan" \
      --output "$output" \
      --anchor-role city_pre_d1 \
      --target-retrieval-surface-variant rank_before_city_compact \
      --behavior-target-grammar-class adjacent_rank_before_city \
      --allow-selection-scope-bank-transfer \
      --evaluation-split all \
      --conditions $conditions \
      --reference-results "$REFERENCE" \
      --reference-condition clean \
      --reference-behavior-outcome correct_next_needle \
      --limit 10 \
      --anchor-sampling seed_first \
      --max-new-tokens 256 \
      --decode-head-ablation-steps 0 2>&1 | tee -a "$LOG"
  echo "COMPLETE ${bank}_at_citypre utc=$(date -u +%FT%TZ)" | tee -a "$LOG"
}

run_bank p0bank "$P0_PLAN" "selected_bank"
run_bank p2bank "$P2_PLAN" "selected_bank"
run_bank cityprebank "$CITYPRE_PLAN" "clean selected_bank"

"$PYTHON" - "$RUN_ROOT" "$P0_PLAN" "$P2_PLAN" "$CITYPRE_PLAN" "$COMPLETE" <<'PY'
import csv
import datetime
import json
import pathlib
import sys

run_root = pathlib.Path(sys.argv[1])
plan_paths = {
    "p0bank": pathlib.Path(sys.argv[2]),
    "p2bank": pathlib.Path(sys.argv[3]),
    "cityprebank": pathlib.Path(sys.argv[4]),
}
complete = pathlib.Path(sys.argv[5])


def plan_info(path):
    with path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 1, (path, len(rows))
    row = rows[0]
    heads = {
        tuple(int(value) for value in head)
        for head in json.loads(row["heads"])
    }
    return row, heads


def transition_key(row):
    return (
        str(row["request_id"]),
        int(row["from_occurrence"]),
        int(row["to_occurrence"]),
    )


plans = {bank: plan_info(path) for bank, path in plan_paths.items()}
cells = []
transition_sets = []
for bank in ("p0bank", "p2bank", "cityprebank"):
    output = run_root / (
        f"head_behavior_single_token_adjbefore_compact_{bank}_at_citypre_k128_v1"
    )
    rows = []
    for path in sorted((output / "shards").glob("trial_*.jsonl")):
        with path.open(encoding="utf-8") as handle:
            rows.extend(json.loads(line) for line in handle if line.strip())
    expected = 20 if bank == "cityprebank" else 10
    assert len(rows) == expected, (bank, len(rows))
    selected = [row for row in rows if row["condition"] == "selected_bank"]
    clean = [row for row in rows if row["condition"] == "clean"]
    assert len(selected) == 10
    assert len(clean) == (10 if bank == "cityprebank" else 0)
    keys = {transition_key(row) for row in selected}
    assert len(keys) == 10
    transition_sets.append(keys)
    for row in selected:
        assert row["intervention_anchor_roles"] == ["city_pre_d1"]
        assert int(row["branch_to_target_token_distance"]) == 0
        assert int(row["head_ablation_decode_steps_requested"]) == 0
        assert int(row["head_ablation_decode_steps_observed"]) == 0
        assert row["head_ablation_decode_policy"] == "prefill_only"
        assert float(row["head_ablation_selected_post_zero_max_abs"]) == 0.0
    failures = sum(
        str(row.get("behavior_outcome")) != "correct_next_needle"
        for row in selected
    )
    cells.append(
        {
            "bank": bank,
            "selection_anchor_role": plans[bank][0]["selection_anchor_role"],
            "intervention_anchor_role": "city_pre_d1",
            "anchors": 10,
            "failures": failures,
            "failure_rate": failures / 10,
            "clean_failures": sum(
                str(row.get("behavior_outcome")) != "correct_next_needle"
                for row in clean
            ),
            "bank_sha256": plans[bank][0]["bank_sha256"],
            "output": str(output),
        }
    )

assert all(keys == transition_sets[0] for keys in transition_sets[1:])
overlaps = {}
for left, right in (
    ("p0bank", "p2bank"),
    ("p0bank", "cityprebank"),
    ("p2bank", "cityprebank"),
):
    left_heads = plans[left][1]
    right_heads = plans[right][1]
    intersection = len(left_heads & right_heads)
    overlaps[f"{left}__{right}"] = {
        "intersection": intersection,
        "jaccard": intersection / len(left_heads | right_heads),
    }

payload = {
    "schema_version": "realistic_niah_v5_qwen_citypre_single_token_bank_scan_v1",
    "status": "PASS",
    "completed_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
    "model_label": "Qwen3-8B",
    "target_grammar_class": "adjacent_rank_before_city",
    "target_retrieval_surface_variant": "rank_before_city_compact",
    "bank_size": 128,
    "paired_transition_count": len(transition_sets[0]),
    "intervention_temporal_scope": "exact_city_pre_d1_prefill_token_only",
    "cells": cells,
    "bank_overlaps": overlaps,
}
complete.write_text(
    json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
)
print(json.dumps(payload, sort_keys=True))
PY

echo "ALL_COMPLETE utc=$(date -u +%FT%TZ)" | tee -a "$LOG"
