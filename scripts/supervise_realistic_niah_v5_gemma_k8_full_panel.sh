#!/usr/bin/env bash
set -euo pipefail

CODE_ROOT="${CODE_ROOT:-/home/ubuntu/CoT-Native-thinking-v5/code/Realistic_CoT_NiaH_Count_native_v5_20260818}"
RUN_ROOT="${RUN_ROOT:-/home/ubuntu/CoT-Native-thinking-v5/runs/v5_native_mechanism_reboot_20260818/Gemma4-E4B}"
PYTHON="${PYTHON:-/home/ubuntu/CoT-Native-thinking-v5/venv/bin/python}"
CACHE_DIR="${CACHE_DIR:-/home/ubuntu/CoT-Native-thinking-v5/cache/huggingface}"
GENERATIONS="$CODE_ROOT/work/v5_trace_parser_v2/Gemma4-E4B_generations_reparsed.jsonl"
CONFIG="$CODE_ROOT/configs/realistic_niah_v5.json"
ROUTING="$CODE_ROOT/configs/realistic_niah_v5_gemma_shared_k8_causal_routes_frozen.json"
PLAN="$RUN_ROOT/causal_plan_p0_samebefore_k8_discovery1234_1253_full300_frozen_v2/retrieval_anchor_bank_plan.csv"
REGISTRY_ROOT="$RUN_ROOT/head_behavior_shared_k8_full300_prompt_balanced_discoverysplit_v2"
REGISTRY="$REGISTRY_ROOT/selected_anchor_registry.jsonl"
OUTPUT="$RUN_ROOT/head_behavior_shared_k8_full300_prompt_balanced_frozenregistry_v3"
EXPECTED_PLAN_SHA="275704a13ac3d2992e6e33208191413af715438342d253569a447b39d4d807ca"
EXPECTED_ROUTING_SHA="69301ed7c28f250fb50ef59472917e41ad16901bc6003674325aa12b0f1199c6"
EXPECTED_REGISTRY_SHA="021f1e5d3b95f232c0cf69c08236b592dbf2f54e8e04df6abaa27c097a0a96f8"
EXPECTED_ANCHORS=270
EXPECTED_TRIALS=1350

mkdir -p "$RUN_ROOT/logs" "$RUN_ROOT/locks"
exec 9>"$RUN_ROOT/locks/gemma_targeted_retrieval_k8_full_panel.lock"
if ! flock -n 9; then
  echo "another Gemma K8 full-panel supervisor owns the lock" >&2
  exit 75
fi

check_sha() {
  local path="$1"
  local expected="$2"
  local actual
  actual="$(sha256sum "$path" | awk '{print $1}')"
  if [[ "$actual" != "$expected" ]]; then
    echo "sha256 mismatch: $path actual=$actual expected=$expected" >&2
    exit 1
  fi
}

test -x "$PYTHON"
test -f "$GENERATIONS"
test -f "$CONFIG"
test -f "$ROUTING"
test -f "$PLAN"
test -f "$REGISTRY"
check_sha "$PLAN" "$EXPECTED_PLAN_SHA"
check_sha "$ROUTING" "$EXPECTED_ROUTING_SHA"
check_sha "$REGISTRY" "$EXPECTED_REGISTRY_SHA"

"$PYTHON" - "$REGISTRY" "$EXPECTED_ANCHORS" <<'PY'
import collections
import json
import pathlib
import sys

registry = pathlib.Path(sys.argv[1])
expected = int(sys.argv[2])
rows = [json.loads(line) for line in registry.open(encoding="utf-8") if line.strip()]
keys = [
    (str(row["request_id"]), int(row["from_occurrence"]), int(row["to_occurrence"]))
    for row in rows
]
counts = collections.Counter(int(row["gold_count"]) for row in rows)
seeds = collections.Counter(int(row["seed"]) for row in rows)
grammars = collections.Counter(str(row["target_grammar_class"]) for row in rows)
confirmation = sum(1254 <= int(row["seed"]) <= 1263 for row in rows)
assert len(rows) == expected, len(rows)
assert len(keys) == len(set(keys)), "duplicate registered transitions"
assert counts == {count: 30 for count in range(2, 11)}, counts
assert seeds == {seed: 9 for seed in range(1234, 1264)}, seeds
assert confirmation == 90, confirmation
assert grammars == {
    "adjacent_rank_after_city": 115,
    "adjacent_rank_before_city": 5,
    "same_unit_rank_after_city": 8,
    "same_unit_rank_before_city": 140,
    "structural_invariant_bullet": 2,
}, grammars
print(json.dumps({
    "anchors": len(rows),
    "confirmation_anchors": confirmation,
    "counts": dict(sorted(counts.items())),
    "grammars": dict(sorted(grammars.items())),
}, sort_keys=True))
PY

echo "START Gemma K8 full panel utc=$(date -u +%FT%TZ)"
(
  cd "$CODE_ROOT"
  HF_HOME="$CACHE_DIR" TOKENIZERS_PARALLELISM=false "$PYTHON" scripts/run_realistic_niah_v5.py causal-heads-behavior \
    --config "$CONFIG" \
    --model Gemma4-E4B \
    --cache-dir "$CACHE_DIR" \
    --device-map auto \
    --torch-dtype bfloat16 \
    --attention-backend sdpa \
    --generations "$GENERATIONS" \
    --plan "$PLAN" \
    --output "$OUTPUT" \
    --anchor-routing "$ROUTING" \
    --behavior-all-routed-grammars \
    --allow-selection-scope-bank-transfer \
    --evaluation-split all \
    --conditions clean selected_bank layer_matched_random \
    --include-secondary \
    --limit 300 \
    --anchor-sampling prompt_balanced \
    --max-new-tokens 256 \
    --decode-head-ablation-steps -1 \
    --anchor-registry-input "$REGISTRY"
) 2>&1 | tee -a "$RUN_ROOT/logs/gemma_targeted_retrieval_k8_full_panel.log"

"$PYTHON" - "$OUTPUT" "$EXPECTED_TRIALS" "$EXPECTED_REGISTRY_SHA" <<'PY'
import collections
import datetime
import hashlib
import json
import pathlib
import sys

output = pathlib.Path(sys.argv[1])
expected_trials = int(sys.argv[2])
expected_registry_sha = sys.argv[3]
manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
files = sorted((output / "shards").glob("trial_*.jsonl"))
rows = []
for path in files:
    with path.open(encoding="utf-8") as handle:
        rows.extend(json.loads(line) for line in handle if line.strip())
conditions = collections.Counter(str(row["condition"]) for row in rows)
registry = output / "selected_anchor_registry.jsonl"
registry_sha = hashlib.sha256(registry.read_bytes()).hexdigest()
assert manifest["evaluation_split"] == "all", manifest["evaluation_split"]
assert int(manifest["scheduled_anchor_count"]) == 270, manifest["scheduled_anchor_count"]
assert int(manifest["completed_shards"]) == expected_trials, manifest["completed_shards"]
assert len(files) == expected_trials, len(files)
assert len(rows) == expected_trials, len(rows)
assert conditions == {
    "clean": 270,
    "selected_bank": 270,
    "layer_matched_random": 810,
}, conditions
assert registry_sha == expected_registry_sha, registry_sha
payload = {
    "schema_version": "realistic_niah_v5_gemma_targeted_retrieval_k8_full_panel_supervisor_v1",
    "status": "PASS",
    "completed_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
    "output": str(output),
    "trials": len(rows),
    "conditions": dict(conditions),
    "registry_sha256": registry_sha,
}
(output.parent / "gemma_targeted_retrieval_k8_full_panel_complete.json").write_text(
    json.dumps(payload, indent=2, sort_keys=True) + "\n",
    encoding="utf-8",
)
print(json.dumps(payload, sort_keys=True))
PY

echo "COMPLETE Gemma K8 full panel utc=$(date -u +%FT%TZ)"
