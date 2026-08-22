#!/usr/bin/env bash
set -euo pipefail

MODEL="${1:?usage: $0 MODEL}"
CODE_ROOT="${CODE_ROOT:-/home/ubuntu/Realistic_CoT_NiaH_Count}"
PYTHON="${PYTHON:-$CODE_ROOT/.venv/bin/python}"
CACHE_DIR="${CACHE_DIR:-$CODE_ROOT/work/hf_cache}"
RUN_ROOT="${RUN_ROOT:-$CODE_ROOT/work/v5_native_count_stream/targeted_count_chain_20d10c_20260821/$MODEL}"
CONFIG="$CODE_ROOT/configs/realistic_niah_v5.json"
DEFAULTS="$CODE_ROOT/configs/realistic_niah_v5_targeted_retrieval_prospective_defaults_v1.json"
PLAN_FREEZER="$CODE_ROOT/scripts/freeze_realistic_niah_v5_targeted_default_plan.py"
PLAN="$RUN_ROOT/frozen_targeted_count_plan.csv"
REGISTRY_RUN="$RUN_ROOT/final_transition_registry"
REGISTRY="$REGISTRY_RUN/selected_anchor_registry.jsonl"
DISCOVERY="$RUN_ROOT/targeted_count_discovery"
DISCOVERY_ANALYSIS="$RUN_ROOT/targeted_count_analysis_discovery"
CONFIRMATION="$RUN_ROOT/targeted_count_confirmation"
CONFIRMATION_ANALYSIS="$RUN_ROOT/targeted_count_analysis_confirmation"
COMPLETE="$RUN_ROOT/targeted_count_complete.json"
LOG="$RUN_ROOT/logs/targeted_count_endpoint.log"

case "$MODEL" in
  Qwen3-8B)
    GENERATIONS="$CODE_ROOT/work/v5_trace_parser_v2/Qwen3-8B_generations_reparsed.jsonl"
    ;;
  Gemma4-E4B)
    GENERATIONS="$CODE_ROOT/work/v5_trace_parser_v2/Gemma4-E4B_generations_reparsed.jsonl"
    ;;
  *)
    echo "unsupported model: $MODEL" >&2
    exit 2
    ;;
esac

mkdir -p "$RUN_ROOT/logs" "$RUN_ROOT/locks" "$CACHE_DIR"
exec 9>"$RUN_ROOT/locks/targeted_count_endpoint.lock"
if ! flock -n 9; then
  echo "another targeted-count supervisor owns $MODEL" >&2
  exit 75
fi

test -x "$PYTHON"
test -f "$CONFIG"
test -f "$GENERATIONS"
test -f "$DEFAULTS"
test -f "$PLAN_FREEZER"

if [[ ! -s "$PLAN" ]]; then
  "$PYTHON" "$PLAN_FREEZER" \
    --root "$CODE_ROOT" \
    --defaults "$DEFAULTS" \
    --model "$MODEL" \
    --output "$PLAN"
fi

PLAN_SIGNATURE="$($PYTHON - "$PLAN" <<'PY'
import csv
import pathlib
import sys

rows = list(csv.DictReader(pathlib.Path(sys.argv[1]).open(encoding="utf-8")))
selected = [row for row in rows if row["condition"] == "selected_bank"]
assert len(selected) == 1, "plan must have exactly one selected bank"
assert "selection_rank" not in rows[0], "selection_rank is prohibited"
print(f"{int(selected[0]['bank_size'])}:{selected[0]['bank_sha256']}")
PY
)"
HISTORICAL_ROOT="$CODE_ROOT/work/v5_native_count_stream/targeted_count_chain_20d10c_20260821/$MODEL"
case "$MODEL:$PLAN_SIGNATURE" in
  Qwen3-8B:128:ef30a8a083468c6e88cb5b0924403884ad758fedbc743de36dd03ab9bc4a742b)
    ROUTING="$CODE_ROOT/configs/realistic_niah_v5_qwen_shared_k128_causal_routes_frozen.json"
    ;;
  Gemma4-E4B:6:2a7652c68454a5333f19324ec5517fe8c22b03ef4955088a283229c8576211b1)
    ROUTING="$CODE_ROOT/configs/realistic_niah_v5_gemma_shared_k6_causal_routes_frozen.json"
    ;;
  Qwen3-8B:125:73aaaeb8f314bd867eff7df43e35d84ca52b60058c25a5a2fa7e8ffafc513659)
    [[ "$RUN_ROOT" == "$HISTORICAL_ROOT" ]] || { echo "K125 is historical-root only" >&2; exit 2; }
    ROUTING="$CODE_ROOT/configs/realistic_niah_v5_qwen_shared_k125_causal_routes_frozen.json"
    ;;
  Gemma4-E4B:8:93a174e36bf14938fdea4a147a032e245caef760f9631e193553f9828d8a6874)
    [[ "$RUN_ROOT" == "$HISTORICAL_ROOT" ]] || { echo "K8 is historical-root only" >&2; exit 2; }
    ROUTING="$CODE_ROOT/configs/realistic_niah_v5_gemma_shared_k8_causal_routes_frozen.json"
    ;;
  *)
    echo "unregistered targeted plan signature: $MODEL:$PLAN_SIGNATURE" >&2
    exit 2
    ;;
esac
test -f "$ROUTING"

run_logged() {
  local label="$1"
  shift
  echo "START $label utc=$(date -u +%FT%TZ)" | tee -a "$LOG"
  "$@" 2>&1 | tee -a "$LOG"
  echo "COMPLETE $label utc=$(date -u +%FT%TZ)" | tee -a "$LOG"
}

behavior_args=(
  --config "$CONFIG"
  --model "$MODEL"
  --cache-dir "$CACHE_DIR"
  --device-map auto
  --torch-dtype bfloat16
  --attention-backend sdpa
  --generations "$GENERATIONS"
  --plan "$PLAN"
  --anchor-routing "$ROUTING"
  --behavior-all-routed-grammars
  --allow-selection-scope-bank-transfer
  --allow-selection-intervention-site-decoupling
  --conditions clean selected_bank layer_matched_random
  --include-secondary
  --counts 2 3 4 5 6 7 8 9 10
  --limit 300
  --max-new-tokens 512
  --decode-head-ablation-steps -1
)

if [[ ! -s "$REGISTRY" ]]; then
  run_logged registry \
    "$PYTHON" "$CODE_ROOT/scripts/run_realistic_niah_v5.py" \
      causal-heads-behavior "${behavior_args[@]}" \
      --output "$REGISTRY_RUN" \
      --evaluation-split all \
      --anchor-sampling prompt_final_transition \
      --freeze-anchor-registry-only
fi

"$PYTHON" - "$REGISTRY" <<'PY'
import collections
import json
import pathlib
import sys

path = pathlib.Path(sys.argv[1])
rows = [json.loads(line) for line in path.open(encoding="utf-8") if line.strip()]
assert rows, "empty final-transition registry"
assert all(int(row["gold_count"]) in range(2, 11) for row in rows)
assert all(int(row["from_occurrence"]) == int(row["gold_count"]) - 1 for row in rows)
assert all(int(row["to_occurrence"]) == int(row["gold_count"]) for row in rows)
seeds = collections.Counter(int(row["seed"]) for row in rows)
assert set(range(1234, 1264)).issubset(seeds), sorted(seeds)
print(json.dumps({"registry_rows": len(rows), "seeds": len(seeds)}, sort_keys=True))
PY

run_logged discovery \
  "$PYTHON" "$CODE_ROOT/scripts/run_realistic_niah_v5.py" \
    causal-heads-behavior "${behavior_args[@]}" \
    --output "$DISCOVERY" \
    --evaluation-split discovery \
    --anchor-registry-input "$REGISTRY"

run_logged analyze_discovery \
  "$PYTHON" "$CODE_ROOT/scripts/analyze_realistic_niah_v5_targeted_count_endpoint.py" \
    --trials "$DISCOVERY" \
    --output "$DISCOVERY_ANALYSIS" \
    --phase discovery

if ! "$PYTHON" - "$DISCOVERY_ANALYSIS/claim_gates.json" <<'PY'
import json
import pathlib
import sys
value = json.loads(pathlib.Path(sys.argv[1]).read_text(encoding="utf-8"))
raise SystemExit(0 if value["targeted_to_count_pass"] else 1)
PY
then
  "$PYTHON" - "$DISCOVERY_ANALYSIS/claim_gates.json" "$COMPLETE" <<'PY'
import datetime
import json
import pathlib
import sys
claims = json.loads(pathlib.Path(sys.argv[1]).read_text(encoding="utf-8"))
payload = {
    "status": "DISCOVERY_NEGATIVE",
    "completed_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
    "confirmation_opened": False,
    "claims": claims,
}
pathlib.Path(sys.argv[2]).write_text(
    json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
)
PY
  echo "DISCOVERY_NEGATIVE $MODEL; confirmation remains sealed" | tee -a "$LOG"
  exit 0
fi

run_logged confirmation \
  "$PYTHON" "$CODE_ROOT/scripts/run_realistic_niah_v5.py" \
    causal-heads-behavior "${behavior_args[@]}" \
    --output "$CONFIRMATION" \
    --evaluation-split confirmation \
    --anchor-registry-input "$REGISTRY"

run_logged analyze_confirmation \
  "$PYTHON" "$CODE_ROOT/scripts/analyze_realistic_niah_v5_targeted_count_endpoint.py" \
    --trials "$CONFIRMATION" \
    --output "$CONFIRMATION_ANALYSIS" \
    --phase confirmation

"$PYTHON" - "$DISCOVERY_ANALYSIS/claim_gates.json" "$CONFIRMATION_ANALYSIS/claim_gates.json" "$COMPLETE" <<'PY'
import datetime
import json
import pathlib
import sys
discovery = json.loads(pathlib.Path(sys.argv[1]).read_text(encoding="utf-8"))
confirmation = json.loads(pathlib.Path(sys.argv[2]).read_text(encoding="utf-8"))
passed = bool(discovery["targeted_to_count_pass"] and confirmation["targeted_to_count_pass"])
payload = {
    "status": "PASS" if passed else "CONFIRMATION_NEGATIVE",
    "completed_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
    "discovery": discovery,
    "confirmation": confirmation,
}
pathlib.Path(sys.argv[3]).write_text(
    json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
)
print(json.dumps({"status": payload["status"]}, sort_keys=True))
PY
