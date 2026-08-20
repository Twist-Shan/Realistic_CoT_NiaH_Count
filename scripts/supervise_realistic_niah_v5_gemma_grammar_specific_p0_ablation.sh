#!/usr/bin/env bash
set -euo pipefail

CODE_ROOT="${CODE_ROOT:-/home/ubuntu/CoT-Native-thinking-v5/code/Realistic_CoT_NiaH_Count_native_v5_p0_20260820}"
OLD_CODE_ROOT="${OLD_CODE_ROOT:-/home/ubuntu/CoT-Native-thinking-v5/code/Realistic_CoT_NiaH_Count_native_v5_20260818}"
QWEN_ROOT="${QWEN_ROOT:-/home/ubuntu/CoT-Native-thinking-v5/runs/v5_native_grammar_specific_p0_20260820/Qwen3-8B}"
RUN_ROOT="${RUN_ROOT:-/home/ubuntu/CoT-Native-thinking-v5/runs/v5_native_grammar_specific_p0_20260820/Gemma4-E4B}"
PYTHON="${PYTHON:-/home/ubuntu/CoT-Native-thinking-v5/venv/bin/python}"
CACHE_DIR="${CACHE_DIR:-/home/ubuntu/CoT-Native-thinking-v5/cache/huggingface}"
CONFIG="$CODE_ROOT/configs/realistic_niah_v5.json"
ROUTING="$CODE_ROOT/configs/realistic_niah_v5_gemma_grammar_specific_p0_causal_routes_frozen.json"
GENERATIONS="$OLD_CODE_ROOT/work/v5_trace_parser_v2/Gemma4-E4B_generations_reparsed.jsonl"
SOURCE_WRITES="$RUN_ROOT/source_attention_p0_all_local_grammars_full_discovery_v1"
SOURCE_COMPLETE="$RUN_ROOT/gemma_grammar_specific_p0_source_complete.json"
REGISTRY="$RUN_ROOT/anchor_registry_all_p0_full300_v1/selected_anchor_registry.jsonl"
REGISTRY_BY_GRAMMAR="$RUN_ROOT/anchor_registry_by_grammar_v1"
LOG="$RUN_ROOT/logs/gemma_grammar_specific_p0_ablation_supervisor.log"
FULL_COMPLETE="$RUN_ROOT/gemma_grammar_specific_p0_k8_full_panel_complete.json"
DOSE_COMPLETE="$RUN_ROOT/gemma_grammar_specific_p0_dose_grid_complete.json"
GRAMMARS=(
  adjacent_rank_after_city
  adjacent_rank_before_city
  same_unit_rank_after_city
  same_unit_rank_before_city
  structural_invariant_bullet
)
DOSES=(1 2 4 6 8)

mkdir -p "$RUN_ROOT/logs" "$RUN_ROOT/locks" "$CACHE_DIR"
exec 9>"$RUN_ROOT/locks/gemma_grammar_specific_p0_ablation.lock"
if ! flock -n 9; then
  echo "another Gemma grammar-specific P0 ablation supervisor owns the lock" >&2
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
  local output
  output="$(plan_dir "$grammar" "$k")"
  if test -f "$output/causal_plan_audit.json" \
    && grep -q "\"registered_bank_size\": $k" "$output/causal_plan_audit.json"; then
    echo "REUSE plan_${grammar}_k${k} utc=$(date -u +%FT%TZ)" | tee -a "$LOG"
    return
  fi
  echo "START plan_${grammar}_k${k}_prefer_layer_matched utc=$(date -u +%FT%TZ)" | tee -a "$LOG"
  set +e
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
      --random-control-matching layer_matched \
      --full-panel-plan 2>&1 | tee -a "$LOG"
  local layer_status="${PIPESTATUS[0]}"
  set -e
  if test "$layer_status" -eq 0; then
    echo "COMPLETE plan_${grammar}_k${k}_layer_matched utc=$(date -u +%FT%TZ)" | tee -a "$LOG"
    return
  fi
  echo "FALLBACK plan_${grammar}_k${k} control=global reason=layer_matched_infeasible utc=$(date -u +%FT%TZ)" | tee -a "$LOG"
  run_logged "plan_${grammar}_k${k}_global" \
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
      --random-control-matching global \
      --full-panel-plan
}

run_grammar_behavior() {
  local grammar="$1"
  local k="$2"
  local split="$3"
  local panel="$4"
  local plan output registry random_condition registered_split_anchors
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
  random_condition="$("$PYTHON" - "$plan/retrieval_anchor_bank_plan.csv" <<'PY'
import csv
import sys

with open(sys.argv[1], encoding="utf-8", newline="") as handle:
    conditions = {
        row["condition"]
        for row in csv.DictReader(handle)
        if row["condition"] != "selected_bank"
    }
assert len(conditions) == 1, conditions
print(next(iter(conditions)))
PY
)"
  run_logged "behavior_${grammar}_k${k}_${panel}" \
    env HF_HOME="$CACHE_DIR" TOKENIZERS_PARALLELISM=false \
    "$PYTHON" scripts/run_realistic_niah_v5.py causal-heads-behavior \
      --config "$CONFIG" \
      --model Gemma4-E4B \
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

# On a shared single-GPU host, wait for Qwen.  A dedicated Gemma host may set
# SKIP_QWEN_WAIT=1; the explicit flag is recorded in the supervisor log.
if test "${SKIP_QWEN_WAIT:-0}" = "1"; then
  echo "SKIP qwen_dose_wait dedicated_gemma_host=1 utc=$(date -u +%FT%TZ)" | tee -a "$LOG"
else
  while ! test -f "$QWEN_ROOT/qwen_grammar_specific_p0_dose_grid_complete.json"; do
    echo "WAIT qwen_dose_complete utc=$(date -u +%FT%TZ)" | tee -a "$LOG"
    sleep 60
  done
fi

cd "$CODE_ROOT"
if test -f "$SOURCE_COMPLETE"; then
  echo "REUSE source_writes_all_p0 completion=$SOURCE_COMPLETE utc=$(date -u +%FT%TZ)" | tee -a "$LOG"
else
run_logged source_writes_all_p0 \
  env HF_HOME="$CACHE_DIR" TOKENIZERS_PARALLELISM=false \
  "$PYTHON" scripts/run_realistic_niah_v5.py causal-source-writes \
    --config "$CONFIG" \
    --model Gemma4-E4B \
    --cache-dir "$CACHE_DIR" \
    --device-map auto \
    --torch-dtype bfloat16 \
    --attention-backend sdpa \
    --generations "$GENERATIONS" \
    --output "$SOURCE_WRITES" \
    --anchor-role p0_item_end \
    --include-secondary

run_logged audit_source_grammar_coverage \
  "$PYTHON" scripts/audit_realistic_niah_v5_source_grammars.py \
    "$SOURCE_WRITES" \
    --output "$RUN_ROOT/gemma_grammar_specific_p0_source_grammar_audit.json"

"$PYTHON" - "$SOURCE_WRITES" "$SOURCE_COMPLETE" <<'PY'
import datetime
import json
import pathlib
import sys

source = pathlib.Path(sys.argv[1])
complete = pathlib.Path(sys.argv[2])
manifest = json.loads((source / "manifest.json").read_text(encoding="utf-8"))
shards = sorted((source / "shards").glob("trial_*.jsonl"))
seeds = set()
for shard in shards:
    with shard.open(encoding="utf-8") as handle:
        first = next(json.loads(line) for line in handle if line.strip())
    seeds.add(int(first["seed"]))
assert seeds == set(range(1234, 1254)), sorted(seeds)
assert int(manifest["completed_shards"]) == len(shards)
payload = {
    "schema_version": "realistic_niah_v5_gemma_p0_full_source_v1",
    "status": "PASS",
    "completed_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
    "completed_shards": len(shards),
    "discovery_seeds": sorted(seeds),
    "source": str(source),
}
complete.write_text(
    json.dumps(payload, indent=2, sort_keys=True) + "\n",
    encoding="utf-8",
)
print(json.dumps(payload, sort_keys=True))
PY
fi

for grammar in "${GRAMMARS[@]}"; do
  build_plan "$grammar" 8
done

FIRST_PLAN="$(plan_dir adjacent_rank_after_city 8)/retrieval_anchor_bank_plan.csv"
run_logged freeze_full_registry \
  env HF_HOME="$CACHE_DIR" TOKENIZERS_PARALLELISM=false \
  "$PYTHON" scripts/run_realistic_niah_v5.py causal-heads-behavior \
    --config "$CONFIG" \
    --model Gemma4-E4B \
    --cache-dir "$CACHE_DIR" \
    --device-map auto \
    --torch-dtype bfloat16 \
    --attention-backend sdpa \
    --generations "$GENERATIONS" \
    --plan "$FIRST_PLAN" \
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
assert {row["target_grammar_class"] for row in rows} == set(grammars)
print(f"REGISTRY_VIEWS_COMPLETE anchors={written} grammars={len(grammars)}")
PY

run_logged clean_full_panel \
  env HF_HOME="$CACHE_DIR" TOKENIZERS_PARALLELISM=false \
  "$PYTHON" scripts/run_realistic_niah_v5.py causal-heads-behavior \
    --config "$CONFIG" \
    --model Gemma4-E4B \
    --cache-dir "$CACHE_DIR" \
    --device-map auto \
    --torch-dtype bfloat16 \
    --attention-backend sdpa \
    --generations "$GENERATIONS" \
    --plan "$FIRST_PLAN" \
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
  run_grammar_behavior "$grammar" 8 all fullpanel_pergrammarregistry_v2
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
summary = []
total_selected = 0
total_random = 0
for grammar in grammars:
    output = (
        run_root
        / f"head_behavior_{grammar}_p0_k8_fullpanel_pergrammarregistry_v2_v1"
    )
    rows = rows_in(output)
    selected = [row for row in rows if row["condition"] == "selected_bank"]
    random_conditions = {
        row["condition"] for row in rows if row["condition"] != "selected_bank"
    }
    assert len(random_conditions) == 1, (grammar, random_conditions)
    random_condition = next(iter(random_conditions))
    random = [row for row in rows if row["condition"] == random_condition]
    assert len(random) == 3 * len(selected), (grammar, len(selected), len(random))
    assert all(row["routed_target_grammar_class"] == grammar for row in rows)
    assert all(row["head_ablation_decode_steps_requested"] == -1 for row in rows)
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
            "random": len(random_split),
            "random_condition": random_condition,
            "random_failures": sum(
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
            "random_condition": random_condition,
            "random_failures": sum(
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
    "schema_version": "realistic_niah_v5_gemma_grammar_specific_p0_full_v1",
    "status": "PASS",
    "completed_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
    "model_label": "Gemma4-E4B",
    "bank_size": 8,
    "selection_anchor_role": "p0_item_end",
    "selection_metric": "target_source_attention_mass",
    "selection_aggregation": "equal_seed_mean_of_within_seed_event_means",
    "selection_scope": "within_target_grammar",
    "random_control_matching": "prefer_exact_layer_matched_else_global_same_k",
    "persistent_ablation": True,
    "registry_rows": len(registry_rows),
    "registry_sha256": hashlib.sha256(registry.read_bytes()).hexdigest(),
    "clean_rows": len(clean_rows),
    "clean_failures": sum(
        row["behavior_outcome"] != "correct_next_needle" for row in clean_rows
    ),
    "selected_rows": total_selected,
    "random_rows": total_random,
    "grammars": summary,
}
complete.write_text(
    json.dumps(payload, indent=2, sort_keys=True) + "\n",
    encoding="utf-8",
)
print(json.dumps(payload, sort_keys=True))
PY

for k in 1 2 4 6; do
  for grammar in "${GRAMMARS[@]}"; do
    build_plan "$grammar" "$k"
    run_grammar_behavior \
      "$grammar" "$k" confirmation confirmation_pergrammarregistry_v2
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
doses = [1, 2, 4, 6, 8]


def rows_in(output):
    rows = []
    for path in sorted((output / "shards").glob("trial_*.jsonl")):
        with path.open(encoding="utf-8") as handle:
            rows.extend(json.loads(line) for line in handle if line.strip())
    return rows


summary = []
for k in doses:
    for grammar in grammars:
        panel = (
            "fullpanel_pergrammarregistry_v2"
            if k == 8
            else "confirmation_pergrammarregistry_v2"
        )
        output = run_root / f"head_behavior_{grammar}_p0_k{k}_{panel}_v1"
        rows = [row for row in rows_in(output) if row["split"] == "confirmation"]
        selected = [row for row in rows if row["condition"] == "selected_bank"]
        random_conditions = {
            row["condition"]
            for row in rows
            if row["condition"] != "selected_bank"
        }
        assert len(random_conditions) == 1, (k, grammar, random_conditions)
        random_condition = next(iter(random_conditions))
        random = [row for row in rows if row["condition"] == random_condition]
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
    "schema_version": "realistic_niah_v5_gemma_grammar_specific_p0_dose_v1",
    "status": "PASS",
    "completed_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
    "model_label": "Gemma4-E4B",
    "doses": doses,
    "selection_scope": "within_target_grammar",
    "selection_anchor_role": "p0_item_end",
    "persistent_ablation": True,
    "random_control_matching": "prefer_exact_layer_matched_else_global_same_k",
    "reporting_policy": {
        "primary_dose_split": "confirmation",
        "discovery_reported_separately_at_registered_primary_k": 8,
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
