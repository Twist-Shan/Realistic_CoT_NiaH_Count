#!/usr/bin/env bash
set -euo pipefail

CODE_ROOT="${CODE_ROOT:-/home/ubuntu/CoT-Native-thinking-v5/code/Realistic_CoT_NiaH_Count_native_v5_p0_20260820}"
OLD_RUN_ROOT="${OLD_RUN_ROOT:-/home/ubuntu/CoT-Native-thinking-v5/runs/v5_native_mechanism_reboot_20260818/Qwen3-8B}"
RUN_ROOT="${RUN_ROOT:-/home/ubuntu/CoT-Native-thinking-v5/runs/v5_native_read_write_stage_20260820/Qwen3-8B}"
PYTHON="${PYTHON:-/home/ubuntu/CoT-Native-thinking-v5/venv/bin/python}"
CACHE_DIR="${CACHE_DIR:-/home/ubuntu/CoT-Native-thinking-v5/cache/huggingface}"
CONFIG="$CODE_ROOT/configs/realistic_niah_v5.json"
GENERATIONS="$CODE_ROOT/work/v5_trace_parser_v2/Qwen3-8B_generations_reparsed.jsonl"
P0_SOURCE="$OLD_RUN_ROOT/source_attention_p0_adjbefore_compact_dev_v1"
P2_SOURCE="$OLD_RUN_ROOT/source_attention_postmarker_adjbefore_compact_dev_v1"
REFERENCE="$OLD_RUN_ROOT/head_behavior_attention_abs_postmarker_adjbefore_compact_persistent_k32_screen10_dev_v1"
P0_PLAN="$RUN_ROOT/causal_plan_single_token_adjbefore_compact_p0_k128_v1"
P2_PLAN="$RUN_ROOT/causal_plan_single_token_adjbefore_compact_p2_k128_v1"
LOG="$RUN_ROOT/logs/qwen_adjbefore_p0_p2_single_token_2x2_k128_supervisor.log"
COMPLETE="$RUN_ROOT/qwen_adjbefore_p0_p2_single_token_2x2_k128_complete.json"

mkdir -p "$RUN_ROOT/logs" "$RUN_ROOT/locks" "$CACHE_DIR"
exec 9>"$RUN_ROOT/locks/qwen_adjbefore_p0_p2_single_token_2x2_k128.lock"
if ! flock -n 9; then
  echo "another Qwen P0/P2 single-token 2x2 owns the lock" >&2
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
  local output="$RUN_ROOT/head_behavior_single_token_2x2_adjbefore_compact_${bank_label}_at_${site_label}_k128_v1"
  local transfer=()
  if [[ "$bank_label" != "$site_label" ]]; then
    transfer=(--allow-selection-scope-bank-transfer)
  fi
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
      --decode-head-ablation-steps 0
}

cd "$CODE_ROOT"
build_plan "$P0_SOURCE" p0_item_end "$P0_PLAN"
build_plan "$P2_SOURCE" post_marker "$P2_PLAN"

# Clean is rerun once at each branch site.  Selected interventions affect only
# the registered prefill token; all later cached decode forwards are untouched.
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
        "head_behavior_single_token_2x2_adjbefore_compact_"
        f"{bank}_at_{site}_k128_v1"
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


def generated_ids(row):
    return tuple(int(value) for value in row["generated_token_ids"])


p0_heads, p0_plan_row = plan_heads(p0_plan)
p2_heads, p2_plan_row = plan_heads(p2_plan)
intersection = len(p0_heads & p2_heads)
union = len(p0_heads | p2_heads)

loaded = {}
for bank, site in (
    ("p0bank", "p0"),
    ("p0bank", "p2"),
    ("p2bank", "p0"),
    ("p2bank", "p2"),
):
    loaded[(bank, site)] = load_output(bank, site)

clean_rows_by_site = {}
for bank, site in (("p0bank", "p0"), ("p2bank", "p2")):
    _output, rows = loaded[(bank, site)]
    clean = [row for row in rows if str(row["condition"]) == "clean"]
    assert len(clean) == 10, (bank, site, len(clean))
    clean_rows_by_site[site] = {transition_key(row): row for row in clean}
    assert len(clean_rows_by_site[site]) == 10

cells = []
selected_transition_sets = []
for bank, site, expected_conditions, expected_role in (
    ("p0bank", "p0", {"clean", "selected_bank"}, "p0_item_end"),
    ("p0bank", "p2", {"selected_bank"}, "post_marker"),
    ("p2bank", "p0", {"selected_bank"}, "p0_item_end"),
    ("p2bank", "p2", {"clean", "selected_bank"}, "post_marker"),
):
    output, rows = loaded[(bank, site)]
    assert {str(row["condition"]) for row in rows} == expected_conditions
    by_condition = {}
    for condition in sorted(expected_conditions):
        selected = [row for row in rows if str(row["condition"]) == condition]
        assert len(selected) == 10, (bank, site, condition, len(selected))
        keys = {transition_key(row) for row in selected}
        assert len(keys) == 10
        failures = sum(
            str(row.get("behavior_outcome")) != "correct_next_needle"
            for row in selected
        )
        by_condition[condition] = {
            "anchors": len(selected),
            "failures": failures,
            "failure_rate": failures / len(selected),
        }
        if condition == "selected_bank":
            selected_transition_sets.append(keys)
            clean_by_key = clean_rows_by_site[site]
            path_preserved = 0
            failures_with_path_preserved = 0
            registered_city_failures = 0
            for row in selected:
                assert row["head_ablation_decode_steps_requested"] == 0
                assert row["head_ablation_decode_steps_observed"] == 0
                assert row["head_ablation_decode_policy"] == "prefill_only"
                assert row["head_ablation_prefill_only"] is True
                assert row["head_ablation_decode_steps_untouched"] is True
                assert row["intervention_site_count"] == 1
                assert float(row["head_ablation_selected_post_zero_max_abs"]) == 0.0
                assert expected_role in {
                    str(value)
                    for value in row.get("intervention_anchor_roles", [])
                }
                key = transition_key(row)
                clean_row = clean_by_key[key]
                offset = int(row["branch_to_target_token_distance"])
                assert offset == int(clean_row["branch_to_target_token_distance"])
                preserved = generated_ids(row)[:offset] == generated_ids(clean_row)[:offset]
                path_preserved += int(preserved)
                failed = str(row.get("behavior_outcome")) != "correct_next_needle"
                failures_with_path_preserved += int(failed and preserved)
                registered_city_failures += int(
                    not bool(row["generated_target_city_exact_at_registered_path_offset"])
                )
            by_condition[condition].update(
                {
                    "pre_city_path_preserved_vs_clean": path_preserved,
                    "behavior_failures_with_pre_city_path_preserved": (
                        failures_with_path_preserved
                    ),
                    "registered_city_token_failures": registered_city_failures,
                    "branch_to_city_distance_tokens": sorted(
                        {
                            int(row["branch_to_target_token_distance"])
                            for row in selected
                        }
                    ),
                }
            )
    cells.append(
        {
            "bank": bank,
            "intervention_site": site,
            "output": str(output),
            "conditions": by_condition,
        }
    )

assert all(keys == selected_transition_sets[0] for keys in selected_transition_sets[1:])
payload = {
    "schema_version": "realistic_niah_v5_qwen_p0_p2_single_token_2x2_v1",
    "status": "PASS",
    "completed_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
    "model_label": "Qwen3-8B",
    "target_grammar_class": "adjacent_rank_before_city",
    "target_retrieval_surface_variant": "rank_before_city_compact",
    "bank_size": 128,
    "selection_metric": "target_source_attention_mass",
    "selection_aggregation": "equal_seed_mean_of_within_seed_event_means",
    "paired_transition_count": len(selected_transition_sets[0]),
    "intervention_temporal_scope": "registered_prefill_token_only",
    "cached_decode_head_outputs_untouched": True,
    "p0_bank_sha256": p0_plan_row["bank_sha256"],
    "p2_bank_sha256": p2_plan_row["bank_sha256"],
    "bank_intersection": intersection,
    "bank_jaccard": intersection / union,
    "interpretation_contract": {
        "p0": (
            "Tests whether selected head output at the previous-item endpoint is "
            "necessary before the next rank marker is generated."
        ),
        "p2": (
            "Tests whether selected head output at the completed next-rank marker "
            "is necessary immediately before city emission."
        ),
        "caveat": (
            "Attention mass is a read/selection proxy and pre-O ablation removes a "
            "head output contribution; the labels read and write are stage-level "
            "interpretations, not literal memory-interface primitives."
        ),
    },
    "cells": cells,
}
complete.write_text(
    json.dumps(payload, indent=2, sort_keys=True) + "\n",
    encoding="utf-8",
)
print(json.dumps(payload, sort_keys=True))
PY

echo "ALL_COMPLETE utc=$(date -u +%FT%TZ)" | tee -a "$LOG"
