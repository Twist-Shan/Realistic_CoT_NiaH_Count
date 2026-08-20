#!/usr/bin/env bash
set -euo pipefail

CODE_ROOT="${CODE_ROOT:-/home/ubuntu/CoT-Native-thinking-v5/code/Realistic_CoT_NiaH_Count_native_v5_p0_20260820}"
RUN_ROOT="${RUN_ROOT:-/home/ubuntu/CoT-Native-thinking-v5/runs/v5_native_unified_p0_20260820/Qwen3-8B}"
PYTHON="${PYTHON:-/home/ubuntu/CoT-Native-thinking-v5/venv/bin/python}"
CACHE_DIR="${CACHE_DIR:-/home/ubuntu/CoT-Native-thinking-v5/cache/huggingface}"
GENERATIONS="$CODE_ROOT/work/v5_trace_parser_v2/Qwen3-8B_generations_reparsed.jsonl"
CONFIG="$CODE_ROOT/configs/realistic_niah_v5.json"
ROUTING="$CODE_ROOT/configs/realistic_niah_v5_qwen_unified_p0_k128_causal_routes_dev.json"
SOURCE_WRITES="$RUN_ROOT/source_attention_p0_all_local_grammars_full_discovery_v2"
SOURCE_COMPLETE="$RUN_ROOT/qwen_unified_p0_full_discovery_source_complete.json"
LOG="$RUN_ROOT/logs/qwen_p0_grammar_specific_k128_discovery_supervisor.log"
COMPLETE="$RUN_ROOT/qwen_p0_grammar_specific_k128_discovery_complete.json"
GRAMMARS=(
  adjacent_rank_after_city
  adjacent_rank_before_city
  same_unit_rank_before_city
  structural_unmarked
  structural_invariant_bullet
  evidence_sequence_unranked
  structural_explicit_rank_before_city
)

mkdir -p "$RUN_ROOT/logs" "$RUN_ROOT/locks" "$CACHE_DIR"
exec 9>"$RUN_ROOT/locks/qwen_p0_grammar_specific_k128_discovery.lock"
if ! flock -n 9; then
  echo "another grammar-specific K128 discovery supervisor owns the lock" >&2
  exit 75
fi

if ! test -f "$SOURCE_COMPLETE"; then
  echo "missing completed full-discovery source capture: $SOURCE_COMPLETE" >&2
  exit 2
fi

grammar_limit() {
  case "$1" in
    structural_invariant_bullet) echo 2 ;;
    evidence_sequence_unranked) echo 1 ;;
    structural_explicit_rank_before_city) echo 1 ;;
    *) echo 20 ;;
  esac
}

run_logged() {
  local label="$1"
  echo "START $label utc=$(date -u +%FT%TZ)" | tee -a "$LOG"
  shift
  "$@" 2>&1 | tee -a "$LOG"
  echo "COMPLETE $label utc=$(date -u +%FT%TZ)" | tee -a "$LOG"
}

cd "$CODE_ROOT"
for grammar in "${GRAMMARS[@]}"; do
  limit="$(grammar_limit "$grammar")"
  plan="$RUN_ROOT/causal_plan_p0_grammar_${grammar}_local_seed_event_k128_discovery_v1"
  output="$RUN_ROOT/head_behavior_p0_grammar_${grammar}_local_k128_discovery_v1"
  run_logged "plan_${grammar}_k128" \
    "$PYTHON" scripts/run_realistic_niah_v5.py causal-plan \
      --config "$CONFIG" \
      --source-writes "$SOURCE_WRITES" \
      --output "$plan" \
      --bank-size 128 \
      --anchor-role p0_item_end \
      --target-grammar-class "$grammar" \
      --selection-metric target_source_attention_mass \
      --selection-eligibility-scope local \
      --selection-aggregation seed_event_mean \
      --full-panel-plan \
      --selected-only-smoke
  run_logged "behavior_${grammar}_k128_limit${limit}" \
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
      --evaluation-split discovery \
      --conditions clean selected_bank \
      --include-secondary \
      --limit "$limit" \
      --anchor-sampling prompt_balanced \
      --max-new-tokens 256 \
      --decode-head-ablation-steps -1
done

"$PYTHON" - "$RUN_ROOT" "$COMPLETE" "${GRAMMARS[@]}" <<'PY'
import datetime
import hashlib
import json
import pathlib
import sys

run_root = pathlib.Path(sys.argv[1])
complete = pathlib.Path(sys.argv[2])
grammars = sys.argv[3:]
limits = {
    "structural_invariant_bullet": 2,
    "evidence_sequence_unranked": 1,
    "structural_explicit_rank_before_city": 1,
}
summary = []
for grammar in grammars:
    limit = limits.get(grammar, 20)
    output = run_root / (
        f"head_behavior_p0_grammar_{grammar}_local_k128_discovery_v1"
    )
    rows = []
    for path in sorted((output / "shards").glob("trial_*.jsonl")):
        with path.open(encoding="utf-8") as handle:
            rows.extend(json.loads(line) for line in handle if line.strip())
    condition_counts = {}
    condition_failures = {}
    outcome_counts = {}
    for condition in ("clean", "selected_bank"):
        selected = [row for row in rows if row["condition"] == condition]
        condition_counts[condition] = len(selected)
        condition_failures[condition] = sum(
            str(row.get("behavior_outcome")) != "correct_next_needle"
            for row in selected
        )
        outcomes = {}
        for row in selected:
            outcome = str(row.get("behavior_outcome"))
            outcomes[outcome] = outcomes.get(outcome, 0) + 1
        outcome_counts[condition] = outcomes
    assert condition_counts == {"clean": limit, "selected_bank": limit}, (
        grammar,
        condition_counts,
    )
    registry = output / "selected_anchor_registry.jsonl"
    plan = run_root / (
        f"causal_plan_p0_grammar_{grammar}_local_seed_event_"
        "k128_discovery_v1"
    )
    audit = json.loads((plan / "causal_plan_audit.json").read_text())
    summary.append(
        {
            "grammar": grammar,
            "anchor_count": limit,
            "selection_seed_ids": audit["selection_scope_seed_coverage"][
                "Qwen3-8B"
            ],
            "selection_seed_count": len(
                audit["selection_scope_seed_coverage"]["Qwen3-8B"]
            ),
            "condition_counts": condition_counts,
            "condition_failures": condition_failures,
            "outcome_counts": outcome_counts,
            "registry_sha256": hashlib.sha256(registry.read_bytes()).hexdigest(),
        }
    )
payload = {
    "schema_version": (
        "realistic_niah_v5_qwen_p0_grammar_specific_k128_discovery_v1"
    ),
    "status": "PASS",
    "completed_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
    "selection_anchor_role": "p0_item_end",
    "selection_eligibility_scope": "local",
    "selection_aggregation": "equal_seed_mean_of_within_seed_event_means",
    "selection_metric": "target_source_attention_mass",
    "bank_size": 128,
    "ablation_schedule": "persistent_pre_o_from_p0_through_decode",
    "grammars": summary,
}
complete.write_text(
    json.dumps(payload, indent=2, sort_keys=True) + "\n",
    encoding="utf-8",
)
print(json.dumps(payload, sort_keys=True))
PY

echo "ALL_COMPLETE utc=$(date -u +%FT%TZ)" | tee -a "$LOG"
