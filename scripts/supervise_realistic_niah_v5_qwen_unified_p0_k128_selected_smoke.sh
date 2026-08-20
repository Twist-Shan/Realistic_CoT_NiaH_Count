#!/usr/bin/env bash
set -euo pipefail

CODE_ROOT="${CODE_ROOT:-/home/ubuntu/CoT-Native-thinking-v5/code/Realistic_CoT_NiaH_Count_native_v5_p0_20260820}"
RUN_ROOT="${RUN_ROOT:-/home/ubuntu/CoT-Native-thinking-v5/runs/v5_native_unified_p0_20260820/Qwen3-8B}"
PYTHON="${PYTHON:-/home/ubuntu/CoT-Native-thinking-v5/venv/bin/python}"
CACHE_DIR="${CACHE_DIR:-/home/ubuntu/CoT-Native-thinking-v5/cache/huggingface}"
GENERATIONS="$CODE_ROOT/work/v5_trace_parser_v2/Qwen3-8B_generations_reparsed.jsonl"
CONFIG="$CODE_ROOT/configs/realistic_niah_v5.json"
ROUTING="$CODE_ROOT/configs/realistic_niah_v5_qwen_unified_p0_k128_causal_routes_dev.json"
SOURCE_WRITES="$RUN_ROOT/source_attention_p0_all_grammars_discovery_v1"
PLAN="$RUN_ROOT/causal_plan_p0_all_grammars_k128_selectedonly_dev_v1"
OUTPUT="$RUN_ROOT/head_behavior_unified_p0_pooledbank_k128_selected_smoke30_v1"
LOG="$RUN_ROOT/logs/qwen_unified_p0_k128_selected_smoke_supervisor.log"
COMPLETE="$RUN_ROOT/qwen_unified_p0_k128_selected_smoke_complete.json"

mkdir -p "$RUN_ROOT/logs" "$RUN_ROOT/locks" "$CACHE_DIR"
exec 9>"$RUN_ROOT/locks/qwen_unified_p0_k128_selected_smoke.lock"
if ! flock -n 9; then
  echo "another unified-P0 Qwen K128 selected smoke owns the lock" >&2
  exit 75
fi

test -x "$PYTHON"
test -f "$GENERATIONS"
test -f "$CONFIG"
test -f "$ROUTING"
test -f "$SOURCE_WRITES/manifest.json"

run_logged() {
  echo "START $1 utc=$(date -u +%FT%TZ)" | tee -a "$LOG"
  shift
  "$@" 2>&1 | tee -a "$LOG"
  echo "COMPLETE utc=$(date -u +%FT%TZ)" | tee -a "$LOG"
}

cd "$CODE_ROOT"
run_logged plan \
  "$PYTHON" scripts/run_realistic_niah_v5.py causal-plan \
    --config "$CONFIG" \
    --source-writes "$SOURCE_WRITES" \
    --output "$PLAN" \
    --bank-size 128 \
    --anchor-role p0_item_end \
    --selection-metric target_source_attention_mass \
    --full-panel-plan \
    --selected-only-smoke

run_logged behavior \
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
    --conditions clean selected_bank \
    --include-secondary \
    --limit 30 \
    --anchor-sampling prompt_balanced \
    --max-new-tokens 256 \
    --decode-head-ablation-steps -1

"$PYTHON" - "$OUTPUT" "$COMPLETE" <<'PY'
import collections
import datetime
import hashlib
import json
import pathlib
import sys

output = pathlib.Path(sys.argv[1])
complete = pathlib.Path(sys.argv[2])
rows = []
for path in sorted((output / "shards").glob("trial_*.jsonl")):
    with path.open(encoding="utf-8") as handle:
        rows.extend(json.loads(line) for line in handle if line.strip())
conditions = collections.Counter(str(row["condition"]) for row in rows)
assert conditions == {"clean": 30, "selected_bank": 30}, conditions

def failed(row):
    return str(row.get("behavior_outcome")) != "correct_next_needle"

clean = [row for row in rows if row["condition"] == "clean"]
selected = [row for row in rows if row["condition"] == "selected_bank"]
grammar = {}
for name in sorted({str(row.get("routed_target_grammar_class")) for row in selected}):
    subset = [
        row
        for row in selected
        if str(row.get("routed_target_grammar_class")) == name
    ]
    grammar[name] = {
        "anchors": len(subset),
        "selected_failures": sum(map(failed, subset)),
        "selected_failure_rate": sum(map(failed, subset)) / len(subset),
    }

manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
registry = output / "selected_anchor_registry.jsonl"
payload = {
    "schema_version": "realistic_niah_v5_qwen_unified_p0_k128_selected_smoke_v1",
    "status": "PASS",
    "completed_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
    "output": str(output),
    "completed_shards": len(rows),
    "conditions": dict(conditions),
    "clean_failure_rate": sum(map(failed, clean)) / len(clean),
    "selected_failures": sum(map(failed, selected)),
    "selected_failure_rate": sum(map(failed, selected)) / len(selected),
    "grammar_failure": grammar,
    "plan_sha256": manifest["plan_sha256"],
    "routing_sha256": manifest["anchor_routing_sha256"],
    "registry_sha256": hashlib.sha256(registry.read_bytes()).hexdigest(),
}
complete.write_text(
    json.dumps(payload, indent=2, sort_keys=True) + "\n",
    encoding="utf-8",
)
print(json.dumps(payload, sort_keys=True))
PY

echo "ALL_COMPLETE utc=$(date -u +%FT%TZ)" | tee -a "$LOG"
