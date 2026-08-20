#!/usr/bin/env bash
set -euo pipefail

CODE_ROOT="${CODE_ROOT:-/home/ubuntu/CoT-Native-thinking-v5/code/Realistic_CoT_NiaH_Count_native_v5_20260818}"
RUN_ROOT="${RUN_ROOT:-/home/ubuntu/CoT-Native-thinking-v5/runs/v5_native_mechanism_reboot_20260818/Qwen3-8B}"
PYTHON="${PYTHON:-/home/ubuntu/CoT-Native-thinking-v5/venv/bin/python}"
CACHE_DIR="${CACHE_DIR:-/home/ubuntu/CoT-Native-thinking-v5/cache/huggingface}"
GENERATIONS="$CODE_ROOT/work/v5_trace_parser_v2/Qwen3-8B_generations_reparsed.jsonl"
CONFIG="$CODE_ROOT/configs/realistic_niah_v5.json"
ROUTING="$CODE_ROOT/configs/realistic_niah_v5_qwen_shared_k125_causal_routes_frozen.json"
SELECTION_CONFIG="$CODE_ROOT/configs/realistic_niah_v5_qwen_shared_k125_full300_selection.json"
K125_OUTPUT="$RUN_ROOT/head_behavior_shared_k125_full300_prompt_balanced_discoverysplit_v2"
REGISTRY="$K125_OUTPUT/selected_anchor_registry.jsonl"
EXPECTED_REGISTRY_SHA="ed75562232fed47312eecc2562c80c825f7b9c48022ee5b27b4e783fc0ccbf12"
EXPECTED_ROUTING_SHA="b7df9c373036b621d1eb6245fee3cee826f1e5bafa4f21feb198ad4d00ab85a8"
EXPECTED_TRIALS=348

declare -A PLAN_SHA=(
  [32]="6b22c13c468d73d3ee4a0dda450b4d29bea910d2fd44dde70eaea26b62c934ee"
  [64]="da2f96d2efe2254290ea89b0be8048992b76d3b0eb9b5376006f56671c5afcc4"
  [80]="4cd6a9efcf1cf50b477a640448b51c23fba1b24765c50ea74664b1b94a2a15b0"
  [96]="099c9559fd7ab49c4b965c2750187c5569421a8081813d79c39f428f51dd2902"
  [112]="3133955b7a2d097dba463d32907d94586e27a2fe75a83157d0ed58e57c669856"
)

mkdir -p "$RUN_ROOT/logs" "$RUN_ROOT/locks"
exec 9>"$RUN_ROOT/locks/qwen_targeted_retrieval_dose_grid.lock"
if ! flock -n 9; then
  echo "another Qwen dose-grid supervisor owns the lock" >&2
  exit 75
fi

test -x "$PYTHON"
test -f "$GENERATIONS"
test -f "$CONFIG"
test -f "$ROUTING"
test -f "$SELECTION_CONFIG"
test -f "$REGISTRY"

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
  printf '%s/causal_plan_shared_compact_k%s_discovery1234_1253_full300_frozen_v2/retrieval_anchor_bank_plan.csv' "$RUN_ROOT" "$k"
}

output_path() {
  local k="$1"
  printf '%s/head_behavior_shared_k%s_confirmation_prompt_balanced_frozenregistry_v2' "$RUN_ROOT" "$k"
}

run_complete() {
  local output="$1"
  "$PYTHON" - "$output/manifest.json" "$EXPECTED_TRIALS" <<'PY'
import json
import pathlib
import sys

manifest_path = pathlib.Path(sys.argv[1])
expected = int(sys.argv[2])
if not manifest_path.is_file():
    raise SystemExit(1)
manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
raise SystemExit(0 if int(manifest.get("completed_shards", -1)) == expected else 1)
PY
}

verify_run() {
  local output="$1"
  "$PYTHON" - "$output" "$EXPECTED_TRIALS" "$EXPECTED_REGISTRY_SHA" <<'PY'
import collections
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
conditions = collections.Counter(str(row.get("condition")) for row in rows)
registry = output / "selected_anchor_registry.jsonl"
registry_sha = hashlib.sha256(registry.read_bytes()).hexdigest()
registry_rows = sum(1 for line in registry.open(encoding="utf-8") if line.strip())
assert manifest["evaluation_split"] == "confirmation", manifest["evaluation_split"]
assert int(manifest["scheduled_anchor_count"]) == 87, manifest["scheduled_anchor_count"]
assert int(manifest["completed_shards"]) == expected_trials, manifest["completed_shards"]
assert len(files) == expected_trials, len(files)
assert len(rows) == expected_trials, len(rows)
assert conditions == {"selected_bank": 87, "layer_matched_random": 261}, conditions
assert registry_rows == 256, registry_rows
assert registry_sha == expected_registry_sha, registry_sha
print(json.dumps({
    "output": str(output),
    "trials": len(rows),
    "conditions": dict(conditions),
    "registry_rows": registry_rows,
    "registry_sha256": registry_sha,
}, sort_keys=True))
PY
}

run_k() {
  local k="$1"
  local plan
  local output
  plan="$(plan_path "$k")"
  output="$(output_path "$k")"
  test -f "$plan"
  check_sha "$plan" "${PLAN_SHA[$k]}"
  echo "START K=$k utc=$(date -u +%FT%TZ)"
  (
    cd "$CODE_ROOT"
    HF_HOME="$CACHE_DIR" TOKENIZERS_PARALLELISM=false "$PYTHON" scripts/run_realistic_niah_v5.py causal-heads-behavior \
      --config "$CONFIG" \
      --model Qwen3-8B \
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
  ) 2>&1 | tee -a "$RUN_ROOT/logs/qwen_targeted_retrieval_k${k}.log"
  verify_run "$output"
  echo "COMPLETE K=$k utc=$(date -u +%FT%TZ)"
}

check_sha "$REGISTRY" "$EXPECTED_REGISTRY_SHA"
check_sha "$ROUTING" "$EXPECTED_ROUTING_SHA"

K96_OUTPUT="$(output_path 96)"
while ! run_complete "$K96_OUTPUT"; do
  if ! pgrep -af "run_realistic_niah_v5.py causal-heads-behavior" | grep -F -- "$K96_OUTPUT" >/dev/null; then
    echo "K96 is incomplete and its worker is not running" >&2
    exit 1
  fi
  sleep 30
done
verify_run "$K96_OUTPUT"

for K in 112 80 64 32; do
  run_k "$K"
done

ANALYSIS="$RUN_ROOT/analysis_shared_targeted_retrieval_registered_v2"
cd "$CODE_ROOT"
"$PYTHON" scripts/analyze_realistic_niah_v5_targeted_retrieval.py \
  --run "32=$(output_path 32)" \
  --run "64=$(output_path 64)" \
  --run "80=$(output_path 80)" \
  --run "96=$K96_OUTPUT" \
  --run "112=$(output_path 112)" \
  --run "125=$K125_OUTPUT" \
  --selection-config "$SELECTION_CONFIG" \
  --output "$ANALYSIS" \
  --bootstrap-samples 10000 \
  2>&1 | tee "$RUN_ROOT/logs/qwen_targeted_retrieval_final_analysis.log"

"$PYTHON" scripts/build_v5_native_targeted_retrieval_report.py \
  --analysis "$ANALYSIS" \
  --selection-config "$SELECTION_CONFIG" \
  --output "$ANALYSIS/targeted_retrieval_report.html" \
  2>&1 | tee "$RUN_ROOT/logs/qwen_targeted_retrieval_final_report.log"

"$PYTHON" - "$RUN_ROOT/qwen_targeted_retrieval_dose_grid_complete.json" "$ANALYSIS" <<'PY'
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
    "schema_version": "realistic_niah_v5_qwen_targeted_retrieval_dose_supervisor_v1",
    "status": "PASS",
    "completed_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
    "bank_sizes": [32, 64, 80, 96, 112, 125],
    "analysis": str(analysis),
    "analysis_manifest_sha256": hashlib.sha256((analysis / "analysis_manifest.json").read_bytes()).hexdigest(),
    "report": str(report),
    "report_sha256": hashlib.sha256(report.read_bytes()).hexdigest(),
}
output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
print(json.dumps(payload, sort_keys=True))
PY

echo "ALL_COMPLETE utc=$(date -u +%FT%TZ)"
