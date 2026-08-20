#!/usr/bin/env bash
set -euo pipefail

CODE_ROOT="${CODE_ROOT:-/home/ubuntu/CoT-Native-thinking-v5/code/Realistic_CoT_NiaH_Count_native_v5_p0_20260820}"
OLD_RUN_ROOT="${OLD_RUN_ROOT:-/home/ubuntu/CoT-Native-thinking-v5/runs/v5_native_mechanism_reboot_20260818/Qwen3-8B}"
RUN_ROOT="${RUN_ROOT:-/home/ubuntu/CoT-Native-thinking-v5/runs/v5_native_unified_p0_20260820/Qwen3-8B}"
PYTHON="${PYTHON:-/home/ubuntu/CoT-Native-thinking-v5/venv/bin/python}"
CACHE_DIR="${CACHE_DIR:-/home/ubuntu/CoT-Native-thinking-v5/cache/huggingface}"
CONFIG="$CODE_ROOT/configs/realistic_niah_v5.json"
GENERATIONS="$CODE_ROOT/work/v5_trace_parser_v2/Qwen3-8B_generations_reparsed.jsonl"
P0_SOURCE="$OLD_RUN_ROOT/source_attention_p0_adjbefore_compact_dev_v1"
P2_SOURCE="$OLD_RUN_ROOT/source_attention_postmarker_adjbefore_compact_dev_v1"
REFERENCE="$OLD_RUN_ROOT/head_behavior_attention_abs_postmarker_adjbefore_compact_persistent_k32_screen10_dev_v1"
P0_PLAN="$RUN_ROOT/causal_plan_2x2_adjbefore_compact_p0_k128_v1"
P2_PLAN="$RUN_ROOT/causal_plan_2x2_adjbefore_compact_p2_k128_v1"
LOG="$RUN_ROOT/logs/qwen_adjbefore_p0_p2_bank_site_2x2_k128_supervisor.log"
COMPLETE="$RUN_ROOT/qwen_adjbefore_p0_p2_bank_site_2x2_k128_complete.json"

mkdir -p "$RUN_ROOT/logs" "$RUN_ROOT/locks" "$CACHE_DIR"
exec 9>"$RUN_ROOT/locks/qwen_adjbefore_p0_p2_bank_site_2x2_k128.lock"
if ! flock -n 9; then
  echo "another Qwen P0/P2 bank-site 2x2 owns the lock" >&2
  exit 75
fi

run_logged() {
  local label="$1"
  echo "START $label utc=$(date -u +%FT%TZ)" | tee -a "$LOG"
  shift
  "$@" 2>&1 | tee -a "$LOG"
  echo "COMPLETE $label utc=$(date -u +%FT%TZ)" | tee -a "$LOG"
}

build_plan() {
  local source="$1"
  local role="$2"
  local output="$3"
  run_logged "plan_${role}_k128" \
    "$PYTHON" scripts/run_realistic_niah_v5.py causal-plan \
      --config "$CONFIG" \
      --source-writes "$source" \
      --output "$output" \
      --bank-size 128 \
      --anchor-role "$role" \
      --target-grammar-class adjacent_rank_before_city \
      --target-retrieval-surface-variant rank_before_city_compact \
      --selection-metric target_source_attention_mass \
      --selection-eligibility-scope local \
      --selection-aggregation seed_event_mean \
      --full-panel-plan \
      --selected-only-smoke
}

run_cell() {
  local bank_label="$1"
  local plan="$2"
  local site_label="$3"
  local role="$4"
  local conditions="$5"
  local output="$RUN_ROOT/head_behavior_2x2_adjbefore_compact_${bank_label}_at_${site_label}_k128_v1"
  local transfer=()
  if [[ "$bank_label" != "$site_label" ]]; then
    transfer=(--allow-selection-scope-bank-transfer)
  fi
  # shellcheck disable=SC2086
  run_logged "behavior_${bank_label}_at_${site_label}_k128" \
    env HF_HOME="$CACHE_DIR" TOKENIZERS_PARALLELISM=false \
    "$PYTHON" scripts/run_realistic_niah_v5.py causal-heads-behavior \
      --config "$CONFIG" \
      --model Qwen3-8B \
      --cache-dir "$CACHE_DIR" \
      --device-map auto \
      --torch-dtype bfloat16 \
      --attention-backend sdpa \
      --generations "$GENERATIONS" \
      --plan "$plan/retrieval_anchor_bank_plan.csv" \
      --output "$output" \
      --anchor-role "$role" \
      --target-retrieval-surface-variant rank_before_city_compact \
      --behavior-target-grammar-class adjacent_rank_before_city \
      "${transfer[@]}" \
      --evaluation-split all \
      --conditions $conditions \
      --reference-results "$REFERENCE" \
      --reference-condition clean \
      --reference-behavior-outcome correct_next_needle \
      --limit 10 \
      --anchor-sampling seed_first \
      --max-new-tokens 256 \
      --decode-head-ablation-steps -1
}

cd "$CODE_ROOT"
build_plan "$P0_SOURCE" p0_item_end "$P0_PLAN"
build_plan "$P2_SOURCE" post_marker "$P2_PLAN"

# Two within-site cells also rerun clean, giving one clean baseline per site.
run_cell p0bank "$P0_PLAN" p0 p0_item_end "clean selected_bank"
run_cell p0bank "$P0_PLAN" p2 post_marker "selected_bank"
run_cell p2bank "$P2_PLAN" p0 p0_item_end "selected_bank"
run_cell p2bank "$P2_PLAN" p2 post_marker "clean selected_bank"

"$PYTHON" - "$RUN_ROOT" "$P0_PLAN" "$P2_PLAN" "$COMPLETE" <<'PY'
import csv
import datetime
import json
import pathlib
import sys

run_root = pathlib.Path(sys.argv[1])
p0_plan = pathlib.Path(sys.argv[2])
p2_plan = pathlib.Path(sys.argv[3])
complete = pathlib.Path(sys.argv[4])


def plan_heads(plan_dir):
    with (plan_dir / "retrieval_anchor_bank_plan.csv").open(
        encoding="utf-8", newline=""
    ) as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 1, len(rows)
    return {
        tuple(int(value) for value in head)
        for head in json.loads(rows[0]["heads"])
    }, rows[0]


def load_output(bank, site):
    output = run_root / (
        f"head_behavior_2x2_adjbefore_compact_{bank}_at_{site}_k128_v1"
    )
    rows = []
    for path in sorted((output / "shards").glob("trial_*.jsonl")):
        with path.open(encoding="utf-8") as handle:
            rows.extend(json.loads(line) for line in handle if line.strip())
    return output, rows


def transition_key(row):
    return (
        str(row["request_id"]),
        int(row["from_occurrence"]),
        int(row["to_occurrence"]),
    )


p0_heads, p0_plan_row = plan_heads(p0_plan)
p2_heads, p2_plan_row = plan_heads(p2_plan)
intersection = len(p0_heads & p2_heads)
union = len(p0_heads | p2_heads)

cells = []
selected_transition_sets = []
clean_by_site = {}
for bank, site, expected_conditions, expected_role in (
    ("p0bank", "p0", {"clean", "selected_bank"}, "p0_item_end"),
    ("p0bank", "p2", {"selected_bank"}, "post_marker"),
    ("p2bank", "p0", {"selected_bank"}, "p0_item_end"),
    ("p2bank", "p2", {"clean", "selected_bank"}, "post_marker"),
):
    output, rows = load_output(bank, site)
    assert {str(row["condition"]) for row in rows} == expected_conditions
    by_condition = {}
    for condition in sorted(expected_conditions):
        selected = [
            row for row in rows if str(row["condition"]) == condition
        ]
        assert len(selected) == 10, (bank, site, condition, len(selected))
        keys = {transition_key(row) for row in selected}
        assert len(keys) == 10
        outcomes = {}
        for row in selected:
            outcome = str(row.get("behavior_outcome"))
            outcomes[outcome] = outcomes.get(outcome, 0) + 1
            assert expected_role in {
                str(value)
                for value in row.get("intervention_anchor_roles", [])
            } or condition == "clean"
            if condition == "selected_bank":
                assert row["head_ablation_decode_steps_requested"] == -1
                assert row["head_ablation_decode_policy"] == (
                    "all_one_token_cached_decode_forwards"
                )
                assert float(row["head_ablation_selected_post_zero_max_abs"]) == 0.0
        failures = sum(
            str(row.get("behavior_outcome")) != "correct_next_needle"
            for row in selected
        )
        by_condition[condition] = {
            "anchors": len(selected),
            "failures": failures,
            "failure_rate": failures / len(selected),
            "outcome_counts": outcomes,
        }
        if condition == "selected_bank":
            selected_transition_sets.append(keys)
        else:
            clean_by_site[site] = by_condition[condition]
    cells.append(
        {
            "bank": bank,
            "ablation_start_site": site,
            "output": str(output),
            "conditions": by_condition,
        }
    )

assert all(
    keys == selected_transition_sets[0] for keys in selected_transition_sets[1:]
)
payload = {
    "schema_version": "realistic_niah_v5_qwen_p0_p2_bank_site_2x2_v1",
    "status": "PASS",
    "completed_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
    "model_label": "Qwen3-8B",
    "target_grammar_class": "adjacent_rank_before_city",
    "target_retrieval_surface_variant": "rank_before_city_compact",
    "bank_size": 128,
    "selection_metric": "target_source_attention_mass",
    "selection_aggregation": "equal_seed_mean_of_within_seed_event_means",
    "paired_transition_count": len(selected_transition_sets[0]),
    "persistent_ablation": True,
    "p0_bank_sha256": p0_plan_row["bank_sha256"],
    "p2_bank_sha256": p2_plan_row["bank_sha256"],
    "bank_intersection": intersection,
    "bank_jaccard": intersection / union,
    "clean_by_site": clean_by_site,
    "cells": cells,
}
complete.write_text(
    json.dumps(payload, indent=2, sort_keys=True) + "\n",
    encoding="utf-8",
)
print(json.dumps(payload, sort_keys=True))
PY

echo "ALL_COMPLETE utc=$(date -u +%FT%TZ)" | tee -a "$LOG"
