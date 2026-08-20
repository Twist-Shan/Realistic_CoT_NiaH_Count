#!/usr/bin/env bash
set -euo pipefail

CODE_ROOT="${CODE_ROOT:-/home/ubuntu/CoT-Native-thinking-v5/code/Realistic_CoT_NiaH_Count_native_v5_p0_20260820}"
RUN_ROOT="${RUN_ROOT:-/home/ubuntu/CoT-Native-thinking-v5/runs/v5_native_unified_p0_20260820/Qwen3-8B}"
PYTHON="${PYTHON:-/home/ubuntu/CoT-Native-thinking-v5/venv/bin/python}"
CACHE_DIR="${CACHE_DIR:-/home/ubuntu/CoT-Native-thinking-v5/cache/huggingface}"
CONFIG="$CODE_ROOT/configs/realistic_niah_v5.json"
GENERATIONS="$CODE_ROOT/work/v5_trace_parser_v2/Qwen3-8B_generations_reparsed.jsonl"
ROUTING="$CODE_ROOT/configs/realistic_niah_v5_qwen_unified_p0_k128_causal_routes_dev.json"
SOURCE_WRITES="$RUN_ROOT/source_attention_p0_all_local_grammars_full_discovery_v2"
REGISTRY="$RUN_ROOT/head_behavior_unified_p0_pooledbank_k128_selected_smoke30_v1/selected_anchor_registry.jsonl"
PLAN="$RUN_ROOT/causal_plan_p0_fullseed_local_k128_selectedonly_retry_v3"
OUTPUT="$RUN_ROOT/head_behavior_unified_p0_fullseed_local_k128_selected_retry30_v3"
LOG="$RUN_ROOT/logs/qwen_unified_p0_fullseed_k128_retry_supervisor.log"
COMPLETE="$RUN_ROOT/qwen_unified_p0_fullseed_k128_retry_complete.json"

mkdir -p "$RUN_ROOT/logs" "$RUN_ROOT/locks" "$CACHE_DIR"
exec 9>"$RUN_ROOT/locks/qwen_unified_p0_fullseed_k128_retry.lock"
if ! flock -n 9; then
  echo "another unified P0 fullseed K128 retry owns the lock" >&2
  exit 75
fi

run_logged() {
  local label="$1"
  echo "START $label utc=$(date -u +%FT%TZ)" | tee -a "$LOG"
  shift
  "$@" 2>&1 | tee -a "$LOG"
  echo "COMPLETE $label utc=$(date -u +%FT%TZ)" | tee -a "$LOG"
}

cd "$CODE_ROOT"
run_logged plan_k128 \
  "$PYTHON" scripts/run_realistic_niah_v5.py causal-plan \
    --config "$CONFIG" \
    --source-writes "$SOURCE_WRITES" \
    --output "$PLAN" \
    --bank-size 128 \
    --anchor-role p0_item_end \
    --selection-metric target_source_attention_mass \
    --selection-eligibility-scope local \
    --selection-aggregation seed_event_mean \
    --full-panel-plan \
    --selected-only-smoke

run_logged behavior_k128 \
  env HF_HOME="$CACHE_DIR" TOKENIZERS_PARALLELISM=false \
  "$PYTHON" scripts/run_realistic_niah_v5.py causal-heads-behavior \
    --config "$CONFIG" \
    --model Qwen3-8B \
    --cache-dir "$CACHE_DIR" \
    --device-map auto \
    --torch-dtype bfloat16 \
    --attention-backend sdpa \
    --generations "$GENERATIONS" \
    --plan "$PLAN/retrieval_anchor_bank_plan.csv" \
    --output "$OUTPUT" \
    --anchor-routing "$ROUTING" \
    --behavior-all-routed-grammars \
    --allow-selection-scope-bank-transfer \
    --evaluation-split discovery \
    --conditions selected_bank \
    --include-secondary \
    --limit 30 \
    --anchor-sampling prompt_balanced \
    --anchor-registry-input "$REGISTRY" \
    --max-new-tokens 256 \
    --decode-head-ablation-steps -1

"$PYTHON" - "$PLAN" "$OUTPUT" "$REGISTRY" "$COMPLETE" <<'PY'
import csv
import datetime
import hashlib
import json
import pathlib
import sys

plan_dir = pathlib.Path(sys.argv[1])
output = pathlib.Path(sys.argv[2])
registry = pathlib.Path(sys.argv[3])
complete = pathlib.Path(sys.argv[4])
rows = []
for path in sorted((output / "shards").glob("trial_*.jsonl")):
    with path.open(encoding="utf-8") as handle:
        rows.extend(json.loads(line) for line in handle if line.strip())
assert len(rows) == 30, len(rows)
assert {row["condition"] for row in rows} == {"selected_bank"}
failures = [
    row
    for row in rows
    if str(row.get("behavior_outcome")) != "correct_next_needle"
]
outcomes = {}
grammars = {}
for row in rows:
    outcome = str(row.get("behavior_outcome"))
    grammar = str(
        row.get("routed_target_grammar_class")
        or row.get("behavior_target_grammar_class")
        or str(row.get("grammar_pair", "unknown")).rsplit(" -> ", 1)[-1]
    )
    outcomes[outcome] = outcomes.get(outcome, 0) + 1
    cell = grammars.setdefault(grammar, {"anchors": 0, "failures": 0})
    cell["anchors"] += 1
    cell["failures"] += outcome != "correct_next_needle"
with (plan_dir / "retrieval_anchor_bank_plan.csv").open(
    encoding="utf-8", newline=""
) as handle:
    plan_rows = list(csv.DictReader(handle))
assert len(plan_rows) == 1
payload = {
    "schema_version": "realistic_niah_v5_qwen_unified_p0_k128_retry_v1",
    "status": "PASS",
    "completed_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
    "selection_anchor_role": "p0_item_end",
    "selection_metric": "target_source_attention_mass",
    "selection_eligibility_scope": "local",
    "selection_aggregation": "equal_seed_mean_of_within_seed_event_means",
    "bank_size": 128,
    "bank_sha256": plan_rows[0]["bank_sha256"],
    "anchor_count": len(rows),
    "registry_sha256": hashlib.sha256(registry.read_bytes()).hexdigest(),
    "selected_failures": len(failures),
    "selected_failure_rate": len(failures) / len(rows),
    "outcome_counts": outcomes,
    "grammar_counts": grammars,
    "ablation_schedule": "persistent_pre_o_from_p0_through_decode",
}
complete.write_text(
    json.dumps(payload, indent=2, sort_keys=True) + "\n",
    encoding="utf-8",
)
print(json.dumps(payload, sort_keys=True))
PY

echo "ALL_COMPLETE utc=$(date -u +%FT%TZ)" | tee -a "$LOG"
