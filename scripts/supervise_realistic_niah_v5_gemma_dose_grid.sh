#!/usr/bin/env bash
set -euo pipefail

CODE_ROOT="${CODE_ROOT:-/home/ubuntu/CoT-Native-thinking-v5/code/Realistic_CoT_NiaH_Count_native_v5_20260818}"
RUN_ROOT="${RUN_ROOT:-/home/ubuntu/CoT-Native-thinking-v5/runs/v5_native_mechanism_reboot_20260818/Gemma4-E4B}"
PYTHON="${PYTHON:-/home/ubuntu/CoT-Native-thinking-v5/venv/bin/python}"
CACHE_DIR="${CACHE_DIR:-/home/ubuntu/CoT-Native-thinking-v5/cache/huggingface}"
GENERATIONS="$CODE_ROOT/work/v5_trace_parser_v2/Gemma4-E4B_generations_reparsed.jsonl"
CONFIG="$CODE_ROOT/configs/realistic_niah_v5.json"
ROUTING="$CODE_ROOT/configs/realistic_niah_v5_gemma_shared_k8_causal_routes_frozen.json"
SELECTION_CONFIG="$CODE_ROOT/configs/realistic_niah_v5_gemma_shared_k8_full300_selection.json"
K8_OUTPUT="$RUN_ROOT/head_behavior_shared_k8_full300_prompt_balanced_frozenregistry_v3"
REGISTRY="$RUN_ROOT/head_behavior_shared_k8_full300_prompt_balanced_discoverysplit_v2/selected_anchor_registry.jsonl"
EXPECTED_REGISTRY_SHA="021f1e5d3b95f232c0cf69c08236b592dbf2f54e8e04df6abaa27c097a0a96f8"
EXPECTED_ROUTING_SHA="69301ed7c28f250fb50ef59472917e41ad16901bc6003674325aa12b0f1199c6"
EXPECTED_TRIALS=360

declare -A PLAN_SHA=(
  [1]="5ef2e7797e4b9547a9f0b59973eb0d5013deec909cf92ddaa38028fe34a5a81c"
  [2]="1e283ec0e0491139eb8f3b0bebd051be0eef275f587edb773eda8289dee9f046"
  [4]="a34e2bc57bf829bc532b590a470ca3a29a9099a714ddb5f0e79867b9b8f3ce6e"
  [6]="348b42d7ff89f9e185d694966b0adc8b7370987cbcea6561bd91a07d1fbee224"
  [8]="275704a13ac3d2992e6e33208191413af715438342d253569a447b39d4d807ca"
)

mkdir -p "$RUN_ROOT/logs" "$RUN_ROOT/locks"
exec 9>"$RUN_ROOT/locks/gemma_targeted_retrieval_dose_grid.lock"
if ! flock -n 9; then
  echo "another Gemma dose-grid supervisor owns the lock" >&2
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

plan_path() {
  local k="$1"
  if [[ "$k" == "8" ]]; then
    printf '%s/causal_plan_p0_samebefore_k8_discovery1234_1253_full300_frozen_v2/retrieval_anchor_bank_plan.csv' "$RUN_ROOT"
  else
    printf '%s/causal_plan_p0_samebefore_k%s_discovery1234_1253_full300_frozen_v3/retrieval_anchor_bank_plan.csv' "$RUN_ROOT" "$k"
  fi
}

output_path() {
  local k="$1"
  printf '%s/head_behavior_shared_k%s_confirmation_prompt_balanced_frozenregistry_v3' "$RUN_ROOT" "$k"
}

test -x "$PYTHON"
test -f "$GENERATIONS"
test -f "$CONFIG"
test -f "$ROUTING"
test -f "$SELECTION_CONFIG"
test -f "$REGISTRY"
test -f "$K8_OUTPUT/manifest.json"
check_sha "$REGISTRY" "$EXPECTED_REGISTRY_SHA"
check_sha "$ROUTING" "$EXPECTED_ROUTING_SHA"
for k in 1 2 4 6 8; do
  check_sha "$(plan_path "$k")" "${PLAN_SHA[$k]}"
done

"$PYTHON" - "$K8_OUTPUT" "$EXPECTED_REGISTRY_SHA" <<'PY'
import collections
import hashlib
import json
import pathlib
import sys

output = pathlib.Path(sys.argv[1])
expected_registry_sha = sys.argv[2]
manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
conditions = collections.Counter()
for path in (output / "shards").glob("trial_*.jsonl"):
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                conditions[json.loads(line)["condition"]] += 1
assert int(manifest["completed_shards"]) == 1350, manifest["completed_shards"]
assert conditions == {
    "clean": 270,
    "selected_bank": 270,
    "layer_matched_random": 810,
}, conditions
assert hashlib.sha256((output / "selected_anchor_registry.jsonl").read_bytes()).hexdigest() == expected_registry_sha
print(json.dumps({"K8_full_panel": "PASS", "conditions": dict(conditions)}, sort_keys=True))
PY

verify_run() {
  local output="$1"
  local k="$2"
  "$PYTHON" - "$output" "$k" "$EXPECTED_TRIALS" "$EXPECTED_REGISTRY_SHA" <<'PY'
import collections
import hashlib
import json
import pathlib
import sys

output = pathlib.Path(sys.argv[1])
k = int(sys.argv[2])
expected_trials = int(sys.argv[3])
expected_registry_sha = sys.argv[4]
manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
files = sorted((output / "shards").glob("trial_*.jsonl"))
rows = []
for path in files:
    with path.open(encoding="utf-8") as handle:
        rows.extend(json.loads(line) for line in handle if line.strip())
conditions = collections.Counter(str(row["condition"]) for row in rows)
registry = output / "selected_anchor_registry.jsonl"
assert manifest["evaluation_split"] == "confirmation", manifest["evaluation_split"]
assert int(manifest["scheduled_anchor_count"]) == 90, manifest["scheduled_anchor_count"]
assert int(manifest["completed_shards"]) == expected_trials, manifest["completed_shards"]
assert len(files) == expected_trials, len(files)
assert len(rows) == expected_trials, len(rows)
assert conditions == {"selected_bank": 90, "layer_matched_random": 270}, conditions
assert sum(1 for line in registry.open(encoding="utf-8") if line.strip()) == 270
assert hashlib.sha256(registry.read_bytes()).hexdigest() == expected_registry_sha
assert {int(row["planned_bank_size"]) for row in rows} == {k}
print(json.dumps({
    "K": k,
    "output": str(output),
    "trials": len(rows),
    "conditions": dict(conditions),
}, sort_keys=True))
PY
}

run_k() {
  local k="$1"
  local plan
  local output
  plan="$(plan_path "$k")"
  output="$(output_path "$k")"
  echo "START K=$k utc=$(date -u +%FT%TZ)"
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
      --plan "$plan" \
      --output "$output" \
      --anchor-routing "$ROUTING" \
      --behavior-all-routed-grammars \
      --allow-selection-scope-bank-transfer \
      --evaluation-split confirmation \
      --conditions selected_bank layer_matched_random \
      --include-secondary \
      --limit 300 \
      --anchor-sampling prompt_balanced \
      --max-new-tokens 256 \
      --decode-head-ablation-steps -1 \
      --anchor-registry-input "$REGISTRY"
  ) 2>&1 | tee -a "$RUN_ROOT/logs/gemma_targeted_retrieval_k${k}.log"
  verify_run "$output" "$k"
  echo "COMPLETE K=$k utc=$(date -u +%FT%TZ)"
}

for k in 1 2 4 6; do
  run_k "$k"
done

ANALYSIS="$RUN_ROOT/analysis_shared_targeted_retrieval_registered_v3"
cd "$CODE_ROOT"
"$PYTHON" scripts/analyze_realistic_niah_v5_targeted_retrieval.py \
  --run "1=$(output_path 1)" \
  --run "2=$(output_path 2)" \
  --run "4=$(output_path 4)" \
  --run "6=$(output_path 6)" \
  --run "8=$K8_OUTPUT" \
  --selection-config "$SELECTION_CONFIG" \
  --output "$ANALYSIS" \
  --bootstrap-samples 10000 \
  2>&1 | tee "$RUN_ROOT/logs/gemma_targeted_retrieval_final_analysis.log"

"$PYTHON" scripts/build_v5_native_targeted_retrieval_report.py \
  --analysis "$ANALYSIS" \
  --selection-config "$SELECTION_CONFIG" \
  --output "$ANALYSIS/targeted_retrieval_report.html" \
  2>&1 | tee "$RUN_ROOT/logs/gemma_targeted_retrieval_final_report.log"

"$PYTHON" - "$RUN_ROOT/gemma_targeted_retrieval_dose_grid_complete.json" "$ANALYSIS" <<'PY'
import datetime
import hashlib
import json
import pathlib
import sys

output = pathlib.Path(sys.argv[1])
analysis = pathlib.Path(sys.argv[2])
manifest = json.loads((analysis / "analysis_manifest.json").read_text(encoding="utf-8"))
assert manifest["analysis_status"] == "complete", manifest["analysis_status"]
report = analysis / "targeted_retrieval_report.html"
assert report.is_file()
payload = {
    "schema_version": "realistic_niah_v5_gemma_targeted_retrieval_dose_supervisor_v1",
    "status": "PASS",
    "completed_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
    "bank_sizes": [1, 2, 4, 6, 8],
    "analysis": str(analysis),
    "analysis_manifest_sha256": hashlib.sha256((analysis / "analysis_manifest.json").read_bytes()).hexdigest(),
    "report": str(report),
    "report_sha256": hashlib.sha256(report.read_bytes()).hexdigest(),
}
output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
print(json.dumps(payload, sort_keys=True))
PY

echo "ALL_COMPLETE utc=$(date -u +%FT%TZ)"
