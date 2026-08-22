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
LOG="$RUN_ROOT/logs/qwen_adjbefore_temporal_window_scan_k128_supervisor.log"
COMPLETE="$RUN_ROOT/qwen_adjbefore_temporal_window_scan_k128_complete.json"

mkdir -p "$RUN_ROOT/logs" "$RUN_ROOT/locks" "$CACHE_DIR"
exec 9>"$RUN_ROOT/locks/qwen_adjbefore_temporal_window_scan_k128.lock"
if ! flock -n 9; then
  echo "another Qwen temporal-window scan owns the lock" >&2
  exit 75
fi

run_window() {
  local bank="$1"
  local plan="$2"
  local site="$3"
  local role="$4"
  local steps="$5"
  local output="$RUN_ROOT/head_behavior_temporal_window_adjbefore_compact_${bank}_at_${site}_d${steps}_k128_v1"
  echo "START ${bank}_at_${site}_d${steps} utc=$(date -u +%FT%TZ)" | tee -a "$LOG"
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
      --anchor-role "$role" \
      --target-retrieval-surface-variant rank_before_city_compact \
      --behavior-target-grammar-class adjacent_rank_before_city \
      --allow-selection-scope-bank-transfer \
      --evaluation-split all \
      --conditions selected_bank \
      --reference-results "$REFERENCE" \
      --reference-condition clean \
      --reference-behavior-outcome correct_next_needle \
      --limit 10 \
      --anchor-sampling seed_first \
      --max-new-tokens 256 \
      --decode-head-ablation-steps "$steps" 2>&1 | tee -a "$LOG"
  echo "COMPLETE ${bank}_at_${site}_d${steps} utc=$(date -u +%FT%TZ)" | tee -a "$LOG"
}

cd "$CODE_ROOT"
test -f "$P0_PLAN"
test -f "$P2_PLAN"

# P0 is 2--4 tokens before the first city token in this frozen cohort.  The
# scan tests whether failure begins before, at, or after the marker boundary.
for steps in 1 2 3 4; do
  run_window p0bank "$P0_PLAN" p0 p0_item_end "$steps"
done

# P2 is one generated boundary token before the first city token.
for steps in 1 2; do
  run_window p2bank "$P2_PLAN" p2 post_marker "$steps"
done

"$PYTHON" - "$RUN_ROOT" "$COMPLETE" <<'PY'
import datetime
import json
import pathlib
import sys

run_root = pathlib.Path(sys.argv[1])
complete = pathlib.Path(sys.argv[2])


def transition_key(row):
    return (
        str(row["request_id"]),
        int(row["from_occurrence"]),
        int(row["to_occurrence"]),
    )


def load(bank, site, steps):
    output = run_root / (
        "head_behavior_temporal_window_adjbefore_compact_"
        f"{bank}_at_{site}_d{steps}_k128_v1"
    )
    rows = []
    for path in sorted((output / "shards").glob("trial_*.jsonl")):
        with path.open(encoding="utf-8") as handle:
            rows.extend(json.loads(line) for line in handle if line.strip())
    assert len(rows) == 10, (bank, site, steps, len(rows))
    assert {str(row["condition"]) for row in rows} == {"selected_bank"}
    assert len({transition_key(row) for row in rows}) == 10
    for row in rows:
        assert int(row["head_ablation_decode_steps_requested"]) == steps
        assert int(row["head_ablation_decode_steps_observed"]) == steps
        assert float(row["head_ablation_selected_post_zero_max_abs"]) == 0.0
    return output, rows


cells = []
transition_sets = []
for bank, site, windows in (
    ("p0bank", "p0", (1, 2, 3, 4)),
    ("p2bank", "p2", (1, 2)),
):
    for steps in windows:
        output, rows = load(bank, site, steps)
        transition_sets.append({transition_key(row) for row in rows})
        failures = [
            row
            for row in rows
            if str(row.get("behavior_outcome")) != "correct_next_needle"
        ]
        by_distance = {}
        for distance in sorted(
            {int(row["branch_to_target_token_distance"]) for row in rows}
        ):
            distance_rows = [
                row
                for row in rows
                if int(row["branch_to_target_token_distance"]) == distance
            ]
            distance_failures = sum(
                str(row.get("behavior_outcome")) != "correct_next_needle"
                for row in distance_rows
            )
            by_distance[str(distance)] = {
                "anchors": len(distance_rows),
                "failures": distance_failures,
                "failure_rate": distance_failures / len(distance_rows),
                "window_reaches_city_query_token": steps >= distance,
            }
        cells.append(
            {
                "bank": bank,
                "start_site": site,
                "decode_steps": steps,
                "output": str(output),
                "anchors": len(rows),
                "failures": len(failures),
                "failure_rate": len(failures) / len(rows),
                "by_branch_to_city_distance": by_distance,
            }
        )

assert all(keys == transition_sets[0] for keys in transition_sets[1:])
payload = {
    "schema_version": "realistic_niah_v5_qwen_temporal_window_scan_v1",
    "status": "PASS",
    "completed_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
    "model_label": "Qwen3-8B",
    "target_grammar_class": "adjacent_rank_before_city",
    "target_retrieval_surface_variant": "rank_before_city_compact",
    "bank_size": 128,
    "paired_transition_count": len(transition_sets[0]),
    "decode_step_semantics": (
        "Number of one-token cached decode forwards additionally ablated after "
        "the registered prefill anchor."
    ),
    "cells": cells,
}
complete.write_text(
    json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
)
print(json.dumps(payload, sort_keys=True))
PY

echo "ALL_COMPLETE utc=$(date -u +%FT%TZ)" | tee -a "$LOG"
