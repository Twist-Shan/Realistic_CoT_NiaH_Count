#!/usr/bin/env bash
set -euo pipefail

CODE_ROOT="${CODE_ROOT:-/home/ubuntu/CoT-Native-thinking-v5/code/Realistic_CoT_NiaH_Count_native_v5_p0_20260820}"
SOURCE_ROOT="${SOURCE_ROOT:-/home/ubuntu/CoT-Native-thinking-v5/runs/v5_native_unified_p0_20260820/Qwen3-8B}"
RUN_ROOT="${RUN_ROOT:-/home/ubuntu/CoT-Native-thinking-v5/runs/v5_native_grammar_specific_p0_20260820/Qwen3-8B}"
PYTHON="${PYTHON:-/home/ubuntu/CoT-Native-thinking-v5/venv/bin/python}"
CACHE_DIR="${CACHE_DIR:-/home/ubuntu/CoT-Native-thinking-v5/cache/huggingface}"
CONFIG="$CODE_ROOT/configs/realistic_niah_v5.json"
ROUTING="$CODE_ROOT/configs/realistic_niah_v5_qwen_grammar_specific_p0_causal_routes_frozen.json"
GENERATIONS="$CODE_ROOT/work/v5_trace_parser_v2/Qwen3-8B_generations_reparsed.jsonl"
SOURCE_WRITES="$SOURCE_ROOT/source_attention_p0_all_local_grammars_full_discovery_v2"
REGISTRY="$RUN_ROOT/anchor_registry_all_p0_full300_v1/selected_anchor_registry.jsonl"
REGISTRY_BY_GRAMMAR="$RUN_ROOT/anchor_registry_by_grammar_v1"
LOG="$RUN_ROOT/logs/qwen_grammar_specific_p0_ablation_supervisor.log"
FULL_COMPLETE="$RUN_ROOT/qwen_grammar_specific_p0_k128_full_panel_complete.json"
DOSE_COMPLETE="$RUN_ROOT/qwen_grammar_specific_p0_dose_grid_complete.json"
GRAMMARS=(
  adjacent_rank_after_city
  adjacent_rank_before_city
  same_unit_rank_before_city
  structural_unmarked
  structural_invariant_bullet
  evidence_sequence_unranked
  structural_explicit_rank_before_city
)
DOSES=(32 64 80 96 112 128)

mkdir -p "$RUN_ROOT/logs" "$RUN_ROOT/locks" "$CACHE_DIR"
exec 9>"$RUN_ROOT/locks/qwen_grammar_specific_p0_ablation.lock"
if ! flock -n 9; then
  echo "another Qwen grammar-specific P0 ablation supervisor owns the lock" >&2
  exit 75
fi

run_logged() {
  local label="$1"
  echo "START $label utc=$(date -u +%FT%TZ)" | tee -a "$LOG"
  shift
  "$@" 2>&1 | tee -a "$LOG"
  echo "COMPLETE $label utc=$(date -u +%FT%TZ)" | tee -a "$LOG"
}

plan_dir() {
  local grammar="$1"
  local k="$2"
  echo "$RUN_ROOT/causal_plan_${grammar}_p0_local_seed_event_k${k}_fullpanel_v1"
}

behavior_dir() {
  local grammar="$1"
  local k="$2"
  local panel="$3"
  echo "$RUN_ROOT/head_behavior_${grammar}_p0_k${k}_${panel}_v1"
}

grammar_registry() {
  local grammar="$1"
  echo "$REGISTRY_BY_GRAMMAR/${grammar}.jsonl"
}

build_plan() {
  local grammar="$1"
  local k="$2"
  local random_matching="${3:-global}"
  local output
  output="$(plan_dir "$grammar" "$k")"
  if test -f "$output/causal_plan_audit.json" \
    && grep -q "\"registered_bank_size\": $k" "$output/causal_plan_audit.json" \
    && grep -q "\"random_control_matching\": \"$random_matching\"" "$output/causal_plan_audit.json"; then
    echo "REUSE plan_${grammar}_k${k}_${random_matching} utc=$(date -u +%FT%TZ)" | tee -a "$LOG"
    return
  fi
  run_logged "plan_${grammar}_k${k}" \
    "$PYTHON" scripts/run_realistic_niah_v5.py causal-plan \
      --config "$CONFIG" \
      --source-writes "$SOURCE_WRITES" \
      --output "$output" \
      --bank-size "$k" \
      --anchor-role p0_item_end \
      --target-grammar-class "$grammar" \
      --selection-metric target_source_attention_mass \
      --selection-eligibility-scope local \
      --selection-aggregation seed_event_mean \
      --random-control-matching "$random_matching" \
      --full-panel-plan
}

run_grammar_behavior() {
  local grammar="$1"
  local k="$2"
  local split="$3"
  local panel="$4"
  local random_condition="${5:-global_random}"
  local plan output registry registered_split_anchors
  plan="$(plan_dir "$grammar" "$k")"
  output="$(behavior_dir "$grammar" "$k" "$panel")"
  registry="$(grammar_registry "$grammar")"
  registered_split_anchors="$("$PYTHON" - "$CONFIG" "$registry" "$split" <<'PY'
import json
import sys

config = json.load(open(sys.argv[1], encoding="utf-8"))
rows = [
    json.loads(line)
    for line in open(sys.argv[2], encoding="utf-8")
    if line.strip()
]
split = sys.argv[3]
if split == "all":
    selected = rows
elif split == "confirmation":
    seeds = set(map(int, config["causal_confirmation_seeds"]))
    selected = [row for row in rows if int(row["seed"]) in seeds]
elif split == "discovery":
    seeds = set(map(int, config["causal_development_seeds"]))
    selected = [row for row in rows if int(row["seed"]) in seeds]
else:
    raise ValueError(f"Unknown split: {split}")
print(len(selected))
PY
)"
  if test "$registered_split_anchors" -eq 0; then
    echo "SKIP behavior_${grammar}_k${k}_${panel} reason=empty_${split}_registry_subset utc=$(date -u +%FT%TZ)" | tee -a "$LOG"
    return
  fi
  run_logged "behavior_${grammar}_k${k}_${panel}" \
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
      --anchor-routing "$ROUTING" \
      --behavior-target-grammar-class "$grammar" \
      --evaluation-split "$split" \
      --conditions selected_bank "$random_condition" \
      --include-secondary \
      --limit 300 \
      --anchor-sampling prompt_balanced \
      --anchor-registry-input "$registry" \
      --max-new-tokens 256 \
      --decode-head-ablation-steps -1
}

cd "$CODE_ROOT"

# Primary K=128 plans are frozen before any new behavioral output is observed.
for grammar in "${GRAMMARS[@]}"; do
  build_plan "$grammar" 128
done

# Freeze a single all-P0 full-panel transition registry.  The pooled plan is
# used only to enumerate anchors; no pooled bank enters a treatment arm.
POOLED_PLAN="$SOURCE_ROOT/causal_plan_p0_fullseed_local_k128_selectedonly_retry_v3/retrieval_anchor_bank_plan.csv"
if test -f "$REGISTRY" && test "$(wc -l < "$REGISTRY")" -eq 256; then
  echo "REUSE freeze_full_registry anchors=256 utc=$(date -u +%FT%TZ)" | tee -a "$LOG"
else
run_logged freeze_full_registry \
  env HF_HOME="$CACHE_DIR" TOKENIZERS_PARALLELISM=false \
  "$PYTHON" scripts/run_realistic_niah_v5.py causal-heads-behavior \
    --config "$CONFIG" \
    --model Qwen3-8B \
    --cache-dir "$CACHE_DIR" \
    --device-map auto \
    --torch-dtype bfloat16 \
    --attention-backend sdpa \
    --generations "$GENERATIONS" \
    --plan "$POOLED_PLAN" \
    --output "$RUN_ROOT/anchor_registry_all_p0_full300_v1" \
    --anchor-routing "$ROUTING" \
    --behavior-all-routed-grammars \
    --allow-selection-scope-bank-transfer \
    --evaluation-split all \
    --conditions clean \
    --include-secondary \
    --limit 300 \
    --anchor-sampling prompt_balanced \
    --freeze-anchor-registry-only \
    --max-new-tokens 256 \
    --decode-head-ablation-steps -1
fi

mkdir -p "$REGISTRY_BY_GRAMMAR"
"$PYTHON" - "$REGISTRY" "$REGISTRY_BY_GRAMMAR" "${GRAMMARS[@]}" <<'PY'
import json
import pathlib
import sys

registry = pathlib.Path(sys.argv[1])
output = pathlib.Path(sys.argv[2])
grammars = sys.argv[3:]
rows = [
    json.loads(line)
    for line in registry.read_text(encoding="utf-8").splitlines()
    if line.strip()
]
written = 0
for grammar in grammars:
    selected = [row for row in rows if row["target_grammar_class"] == grammar]
    path = output / f"{grammar}.jsonl"
    path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in selected),
        encoding="utf-8",
    )
    written += len(selected)
    print(f"REGISTRY_VIEW grammar={grammar} anchors={len(selected)} path={path}")
assert written == len(rows), (written, len(rows))
assert {
    row["target_grammar_class"] for row in rows
} == set(grammars)
print(f"REGISTRY_VIEWS_COMPLETE anchors={written} grammars={len(grammars)}")
PY

run_logged clean_full_panel \
  env HF_HOME="$CACHE_DIR" TOKENIZERS_PARALLELISM=false \
  "$PYTHON" scripts/run_realistic_niah_v5.py causal-heads-behavior \
    --config "$CONFIG" \
    --model Qwen3-8B \
    --cache-dir "$CACHE_DIR" \
    --device-map auto \
    --torch-dtype bfloat16 \
    --attention-backend sdpa \
    --generations "$GENERATIONS" \
    --plan "$POOLED_PLAN" \
    --output "$RUN_ROOT/head_behavior_clean_all_p0_full300_v1" \
    --anchor-routing "$ROUTING" \
    --behavior-all-routed-grammars \
    --allow-selection-scope-bank-transfer \
    --evaluation-split all \
    --conditions clean \
    --include-secondary \
    --limit 300 \
    --anchor-sampling prompt_balanced \
    --anchor-registry-input "$REGISTRY" \
    --max-new-tokens 256 \
    --decode-head-ablation-steps -1

for grammar in "${GRAMMARS[@]}"; do
  run_grammar_behavior "$grammar" 128 all fullpanel_pergrammarregistry_v2
done

"$PYTHON" - "$RUN_ROOT" "$REGISTRY" "$FULL_COMPLETE" "${GRAMMARS[@]}" <<'PY'
import datetime
import hashlib
import json
import pathlib
import sys

run_root = pathlib.Path(sys.argv[1])
registry = pathlib.Path(sys.argv[2])
complete = pathlib.Path(sys.argv[3])
grammars = sys.argv[4:]


def rows_in(output):
    rows = []
    for path in sorted((output / "shards").glob("trial_*.jsonl")):
        with path.open(encoding="utf-8") as handle:
            rows.extend(json.loads(line) for line in handle if line.strip())
    return rows


registry_rows = [
    json.loads(line)
    for line in registry.read_text(encoding="utf-8").splitlines()
    if line.strip()
]
clean_rows = rows_in(run_root / "head_behavior_clean_all_p0_full300_v1")
assert len(clean_rows) == len(registry_rows)
assert {row["condition"] for row in clean_rows} == {"clean"}
summary = []
total_selected = 0
total_random = 0
for grammar in grammars:
    output = (
        run_root
        / f"head_behavior_{grammar}_p0_k128_fullpanel_pergrammarregistry_v2_v1"
    )
    rows = rows_in(output)
    selected = [row for row in rows if row["condition"] == "selected_bank"]
    random = [row for row in rows if row["condition"] == "global_random"]
    assert len(random) == 3 * len(selected), (grammar, len(selected), len(random))
    assert all(row["routed_target_grammar_class"] == grammar for row in rows)
    assert all(row["head_ablation_decode_steps_requested"] == -1 for row in rows)
    assert all(
        float(row["head_ablation_selected_post_zero_max_abs"]) == 0.0
        for row in rows
    )
    total_selected += len(selected)
    total_random += len(random)
    by_split = {}
    for split in ("discovery", "confirmation"):
        selected_split = [row for row in selected if row["split"] == split]
        random_split = [row for row in random if row["split"] == split]
        by_split[split] = {
            "selected": len(selected_split),
            "selected_failures": sum(
                row["behavior_outcome"] != "correct_next_needle"
                for row in selected_split
            ),
            "global_random": len(random_split),
            "global_random_failures": sum(
                row["behavior_outcome"] != "correct_next_needle"
                for row in random_split
            ),
        }
    summary.append(
        {
            "grammar": grammar,
            "anchors": len(selected),
            "selected_failures": sum(
                row["behavior_outcome"] != "correct_next_needle"
                for row in selected
            ),
            "global_random_failures": sum(
                row["behavior_outcome"] != "correct_next_needle"
                for row in random
            ),
            "by_split": by_split,
            "output": str(output),
        }
    )
assert total_selected == len(registry_rows), (total_selected, len(registry_rows))
assert total_random == 3 * len(registry_rows)
payload = {
    "schema_version": "realistic_niah_v5_qwen_grammar_specific_p0_full_v1",
    "status": "PASS",
    "completed_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
    "model_label": "Qwen3-8B",
    "bank_size": 128,
    "selection_anchor_role": "p0_item_end",
    "selection_metric": "target_source_attention_mass",
    "selection_aggregation": "equal_seed_mean_of_within_seed_event_means",
    "selection_scope": "within_target_grammar",
    "random_control_matching": "global_same_k_without_selected_overlap",
    "persistent_ablation": True,
    "registry_rows": len(registry_rows),
    "registry_sha256": hashlib.sha256(registry.read_bytes()).hexdigest(),
    "clean_rows": len(clean_rows),
    "clean_failures": sum(
        row["behavior_outcome"] != "correct_next_needle" for row in clean_rows
    ),
    "selected_rows": total_selected,
    "global_random_rows": total_random,
    "grammars": summary,
}
complete.write_text(
    json.dumps(payload, indent=2, sort_keys=True) + "\n",
    encoding="utf-8",
)
print(json.dumps(payload, sort_keys=True))
PY

# Frozen confirmation dose grid. K=128 is read from the full panel above.
for k in 32 64 80 96 112; do
  for grammar in "${GRAMMARS[@]}"; do
    build_plan "$grammar" "$k" layer_matched
    run_grammar_behavior \
      "$grammar" "$k" confirmation \
      confirmation_pergrammarregistry_v2 layer_matched_random
  done
done

"$PYTHON" - "$RUN_ROOT" "$DOSE_COMPLETE" "${GRAMMARS[@]}" <<'PY'
import datetime
import json
import pathlib
import sys

run_root = pathlib.Path(sys.argv[1])
complete = pathlib.Path(sys.argv[2])
grammars = sys.argv[3:]
doses = [32, 64, 80, 96, 112, 128]


def rows_in(output):
    rows = []
    for path in sorted((output / "shards").glob("trial_*.jsonl")):
        with path.open(encoding="utf-8") as handle:
            rows.extend(json.loads(line) for line in handle if line.strip())
    return rows


summary = []
for k in doses:
    random_condition = (
        "global_random" if k == 128 else "layer_matched_random"
    )
    for grammar in grammars:
        panel = (
            "fullpanel_pergrammarregistry_v2"
            if k == 128
            else "confirmation_pergrammarregistry_v2"
        )
        output = run_root / f"head_behavior_{grammar}_p0_k{k}_{panel}_v1"
        rows = [row for row in rows_in(output) if row["split"] == "confirmation"]
        selected = [row for row in rows if row["condition"] == "selected_bank"]
        random = [
            row for row in rows
            if row["condition"] == random_condition
        ]
        assert len(random) == 3 * len(selected)
        summary.append(
            {
                "bank_size": k,
                "grammar": grammar,
                "confirmation_anchors": len(selected),
                "exploratory": len(selected) < 10,
                "selected_failures": sum(
                    row["behavior_outcome"] != "correct_next_needle"
                    for row in selected
                ),
                "random_condition": random_condition,
                "random_failures": sum(
                    row["behavior_outcome"] != "correct_next_needle"
                    for row in random
                ),
            }
        )
overall = []
for k in doses:
    k_rows = [row for row in summary if row["bank_size"] == k]
    for scope, scoped_rows in (
        ("all_registered_grammars", k_rows),
        (
            "non_exploratory_grammars",
            [row for row in k_rows if not row["exploratory"]],
        ),
    ):
        selected_n = sum(row["confirmation_anchors"] for row in scoped_rows)
        selected_failures = sum(row["selected_failures"] for row in scoped_rows)
        random_n = 3 * selected_n
        random_failures = sum(row["random_failures"] for row in scoped_rows)
        selected_rate = selected_failures / selected_n if selected_n else None
        random_rate = random_failures / random_n if random_n else None
        overall.append(
            {
                "bank_size": k,
                "scope": scope,
                "confirmation_anchors": selected_n,
                "selected_failure_rate": selected_rate,
                "random_failure_rate": random_rate,
                "selected_minus_random_failure_rate": (
                    selected_rate - random_rate
                    if selected_rate is not None and random_rate is not None
                    else None
                ),
            }
        )
payload = {
    "schema_version": "realistic_niah_v5_qwen_grammar_specific_p0_dose_v1",
    "status": "PASS",
    "completed_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
    "model_label": "Qwen3-8B",
    "doses": doses,
    "selection_scope": "within_target_grammar",
    "selection_anchor_role": "p0_item_end",
    "persistent_ablation": True,
    "reporting_policy": {
        "primary_dose_split": "confirmation",
        "discovery_reported_separately_at_registered_primary_k": 128,
        "exploratory_grammar_rule": "confirmation_anchors_lt_10",
        "overall_curve_weighting": "registered_anchor_weighted",
    },
    "rows": summary,
    "overall": overall,
}
complete.write_text(
    json.dumps(payload, indent=2, sort_keys=True) + "\n",
    encoding="utf-8",
)
print(json.dumps(payload, sort_keys=True))
PY

echo "ALL_COMPLETE utc=$(date -u +%FT%TZ)" | tee -a "$LOG"
