#!/usr/bin/env bash
set -euo pipefail

MODEL=${1:?usage: $0 MODEL GPU_INDEX}
GPU_INDEX=${2:?usage: $0 MODEL GPU_INDEX}
ROOT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$ROOT_DIR"

case "$MODEL" in
  Qwen3-8B)
    SOURCE_LAYER=19
    PLAN_SHA=a64ede593478eefbb1d1d5d50e88646d2aa4eeefa824c018d215afac806284f5
    DISCOVERY_SHA=a1936017eb6b29006b3a8818c88ac767ee87748e3b367e121798bc62e5662c1b
    SELECTION="$ROOT_DIR/configs/realistic_niah_v5_qwen_shared_k128_targeted_selection_frozen.json"
    ROUTING="$ROOT_DIR/configs/realistic_niah_v5_qwen_shared_k128_causal_routes_frozen.json"
    BASIS="$ROOT_DIR/work/v5_native_count_stream/representation_20260820/Qwen3-8B/item_end_discovery_basis.npz"
    ;;
  Gemma4-E4B)
    SOURCE_LAYER=16
    PLAN_SHA=97adb5a6f01944d29b4340db5dfed0134653d5cd29c2f8a7efe234b98ad53441
    DISCOVERY_SHA=7fa87d7cece0a3fd5a7d7085e8f1ed5540b38537df08d403f4a2d95d8d460f9b
    SELECTION="$ROOT_DIR/configs/realistic_niah_v5_gemma_shared_k6_targeted_selection_frozen.json"
    ROUTING="$ROOT_DIR/configs/realistic_niah_v5_gemma_shared_k6_causal_routes_frozen.json"
    BASIS="$ROOT_DIR/work/v5_native_count_stream/representation_20260820/Gemma4-E4B/item_end_discovery_basis.npz"
    ;;
  *)
    echo "unsupported model: $MODEL" >&2
    exit 2
    ;;
esac

PYTHON="$ROOT_DIR/.venv/bin/python"
RUNNER="$ROOT_DIR/scripts/run_realistic_niah_v5_count_stream.py"
ANALYZER="$ROOT_DIR/scripts/analyze_realistic_niah_v5_commit_state_to_targeted_query.py"
PROTOCOL="$ROOT_DIR/configs/realistic_niah_v5_commit_state_to_targeted_query_confirmation_v1.json"
MECHANISM="$ROOT_DIR/configs/realistic_niah_v5_native_count_stream_confirmation_v1.json"
V5_CONFIG="$ROOT_DIR/configs/realistic_niah_v5.json"
GENERATIONS="$ROOT_DIR/work/v5_trace_parser_v2/${MODEL}_generations_reparsed.jsonl"
NATIVE_ROOT="$ROOT_DIR/work/v5_native_count_stream/native_loop_chain_k128_k6_20d10c_20260821_v1/$MODEL"
PLAN="$NATIVE_ROOT/plan_confirmation_offsets123/native_loop_plan.csv"
DISCOVERY="$ROOT_DIR/work/v5_native_count_stream/commit_state_to_targeted_query_reanalysis_20d_20260822_v1/$MODEL/analysis_discovery/claim_gates.json"
COUNTER_COMPLETE="$ROOT_DIR/work/v5_native_count_stream/targeted_counter_bridge_k128_k6_20d10c_20260821_v1/$MODEL/targeted_counter_complete.json"
OUTPUT_ROOT="$ROOT_DIR/work/v5_native_count_stream/commit_state_to_targeted_query_20d10c_20260822_v1/$MODEL"
TRIALS="$OUTPUT_ROOT/confirmation"
ANALYSIS="$OUTPUT_ROOT/analysis_confirmation"
COMPLETE="$OUTPUT_ROOT/commit_to_query_complete.json"
LOG="$OUTPUT_ROOT/logs/commit_to_query.log"
mkdir -p "$OUTPUT_ROOT/logs" "$OUTPUT_ROOT/locks"

exec 9>"$OUTPUT_ROOT/locks/commit_to_query.lock"
if ! flock -n 9; then
  echo "another $MODEL commit-to-query supervisor owns the lock" >&2
  exit 3
fi

"$PYTHON" - "$PROTOCOL" "$MODEL" "$PLAN" "$PLAN_SHA" "$DISCOVERY" "$DISCOVERY_SHA" <<'PY'
import hashlib, json, pathlib, sys

protocol_path, model, plan, plan_sha, discovery, discovery_sha = sys.argv[1:]
protocol = json.loads(pathlib.Path(protocol_path).read_text())
assert protocol["status"] == "FROZEN_BEFORE_CONFIRMATION_OUTCOME"
assert protocol["seed_contract"]["discovery"] == list(range(1234, 1254))
assert protocol["seed_contract"]["confirmation"] == list(range(1254, 1264))
assert protocol["seed_contract"]["outcome_blind"] is True
assert protocol["seed_contract"]["selection_rank_used"] is False
assert hashlib.sha256(pathlib.Path(plan).read_bytes()).hexdigest() == plan_sha
assert hashlib.sha256(pathlib.Path(discovery).read_bytes()).hexdigest() == discovery_sha
result = json.loads(pathlib.Path(discovery).read_text())
assert result["analysis_status"] == "POSTHOC_SEALED_DISCOVERY"
assert result["seed_count"] == 20 and result["confirmation_eligible"] is True
assert int(protocol["models"][model]["source_layer"]) in (16, 19)
print(json.dumps({"model": model, "protocol_audit": "PASS"}, sort_keys=True))
PY

echo "WAIT_TARGETED_COUNTER model=$MODEL path=$COUNTER_COMPLETE" | tee -a "$LOG"
while [[ ! -s "$COUNTER_COMPLETE" ]]; do
  sleep 30
done
echo "START_CONFIRMATION model=$MODEL utc=$(date -u +%FT%TZ)" | tee -a "$LOG"

CUDA_VISIBLE_DEVICES="$GPU_INDEX" "$PYTHON" "$RUNNER" p0-native-loop \
  --mechanism-config "$MECHANISM" \
  --v5-config "$V5_CONFIG" \
  --model "$MODEL" \
  --cache-dir "$ROOT_DIR/work/hf_cache" \
  --device-map auto \
  --torch-dtype bfloat16 \
  --attention-backend sdpa \
  --generations "$GENERATIONS" \
  --seed-role confirmation \
  --cohort parser_hit \
  --plan "$PLAN" \
  --basis "$BASIS" \
  --layer "$SOURCE_LAYER" \
  --targeted-selection "$SELECTION" \
  --anchor-routing "$ROUTING" \
  --conditions clean self_patch full_donor_patch count_subspace_transplant norm_matched_orthogonal_patch \
  --donor-offsets -3 -2 -1 1 2 3 \
  --random-seed 20260821 \
  --skip-greedy \
  --output "$TRIALS" 2>&1 | tee -a "$LOG"

"$PYTHON" "$ANALYZER" \
  --trials "$TRIALS" \
  --phase confirmation \
  --bootstrap-samples 10000 \
  --random-seed 20260822 \
  --output "$ANALYSIS" 2>&1 | tee -a "$LOG"

"$PYTHON" - "$DISCOVERY" "$ANALYSIS/claim_gates.json" "$COMPLETE" "$MODEL" <<'PY'
import datetime as dt, json, os, pathlib, sys

discovery = json.loads(pathlib.Path(sys.argv[1]).read_text())
confirmation = json.loads(pathlib.Path(sys.argv[2]).read_text())
directional = bool(
    discovery["directional_signal_pass"]
    and confirmation["directional_signal_pass"]
)
value = {
    "schema_version": "realistic_niah_v5_commit_state_to_targeted_query_complete_v1",
    "status": "PASS" if directional else "CONFIRMATION_NEGATIVE",
    "model_label": sys.argv[4],
    "completed_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
    "discovery_seed_count": 20,
    "confirmation_seed_count": 10,
    "outcome_blind": True,
    "selection_rank_used": False,
    "discovery_is_posthoc_sealed_reanalysis": True,
    "confirmation_is_prospective": True,
    "direct_commit_to_targeted_query_pass": directional,
    "discovery": discovery,
    "confirmation": confirmation,
}
path = pathlib.Path(sys.argv[3])
temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
temporary.replace(path)
print(json.dumps({"status": value["status"]}, sort_keys=True))
PY
