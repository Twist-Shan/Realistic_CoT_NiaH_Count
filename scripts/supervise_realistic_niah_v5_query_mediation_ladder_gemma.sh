#!/usr/bin/env bash
set -euo pipefail

MODEL="${1:-Gemma4-E4B}"
if [[ "$MODEL" != "Gemma4-E4B" ]]; then
  echo "This frozen ladder is registered only for Gemma4-E4B" >&2
  exit 2
fi

REPO="/home/ubuntu/Realistic_CoT_NiaH_Count"
PY="$REPO/.venv/bin/python"
RUNNER="$REPO/scripts/run_realistic_niah_v5_count_stream.py"
ANALYZER="$REPO/scripts/analyze_realistic_niah_v5_query_mediation.py"
ROOT="$REPO/work/v5_native_count_stream/query_mediation_ladder_k6_20d10c_20260821_v1/$MODEL"
NATIVE_ROOT="$REPO/work/v5_native_count_stream/native_loop_chain_k128_k6_20d10c_20260821_v1/$MODEL"
DISCOVERY_PLAN="$NATIVE_ROOT/plan_discovery_offsets123/native_loop_plan.csv"
CONFIRMATION_PLAN="$NATIVE_ROOT/plan_confirmation_offsets123/native_loop_plan.csv"
HEAD_PLAN="$ROOT/head_plan/query_mediation_head_plan.json"
BASIS="$REPO/work/v5_native_count_stream/representation_20260820/$MODEL/item_end_discovery_basis.npz"
SELECTION="$REPO/configs/realistic_niah_v5_gemma_shared_k6_targeted_selection_frozen.json"
ROUTING="$REPO/configs/realistic_niah_v5_gemma_shared_k6_causal_routes_frozen.json"
GENERATIONS="$REPO/work/v5_trace_parser_v2/Gemma4-E4B_generations_reparsed.jsonl"
MECHANISM="$REPO/configs/realistic_niah_v5_native_count_stream_confirmation_v1.json"
V5="$REPO/configs/realistic_niah_v5.json"
LOCK="$ROOT/query_mediation_ladder.lock"

mkdir -p "$ROOT/logs"
exec 9>"$LOCK"
if ! flock -n 9; then
  echo "query-mediation ladder lock conflict: $LOCK" >&2
  exit 3
fi

for required in "$DISCOVERY_PLAN" "$CONFIRMATION_PLAN" "$HEAD_PLAN" "$BASIS" "$SELECTION" "$ROUTING" "$GENERATIONS"; do
  if [[ ! -f "$required" ]]; then
    echo "missing frozen query-mediation input: $required" >&2
    exit 4
  fi
done

export CUDA_VISIBLE_DEVICES=1
export TOKENIZERS_PARALLELISM=false

run_trials() {
  local phase="$1"
  local seed_role="$2"
  local plan="$3"
  local geometry="$4"
  local output="$ROOT/${geometry}_${phase}"
  "$PY" "$RUNNER" p0-query-mediation \
    --mechanism-config "$MECHANISM" \
    --v5-config "$V5" \
    --model "$MODEL" \
    --cache-dir "$REPO/work/hf_cache" \
    --device-map auto \
    --torch-dtype bfloat16 \
    --attention-backend sdpa \
    --generations "$GENERATIONS" \
    --cohort parser_hit \
    --seed-role "$seed_role" \
    --plan "$plan" \
    --basis "$BASIS" \
    --layer 16 \
    --geometry "$geometry" \
    --targeted-selection "$SELECTION" \
    --anchor-routing "$ROUTING" \
    --head-plan "$HEAD_PLAN" \
    --donor-offsets -3 -2 -1 1 2 3 \
    --random-seed 20260821 \
    --skip-greedy \
    --output "$output"
  "$PY" "$ANALYZER" \
    --trials "$output" \
    --phase "$phase" \
    --geometry "$geometry" \
    --bootstrap-samples 10000 \
    --random-seed 20260821 \
    --output "$ROOT/analysis_${geometry}_${phase}"
}

gate_passes() {
  local gate="$1"
  "$PY" -c 'import json,sys; d=json.load(open(sys.argv[1])); sys.exit(0 if d["geometry_pass"] else 1)' "$gate"
}

selected_geometry=""
for geometry in endpoint suffix4 suffix8; do
  run_trials discovery development "$DISCOVERY_PLAN" "$geometry"
  gate="$ROOT/analysis_${geometry}_discovery/claim_gates.json"
  if gate_passes "$gate"; then
    selected_geometry="$geometry"
    break
  fi
done

if [[ -z "$selected_geometry" ]]; then
  "$PY" -c 'import datetime,json,sys; out={"schema_version":"realistic_niah_v5_query_mediation_complete_v1","model_label":"Gemma4-E4B","status":"DISCOVERY_GEOMETRY_EXHAUSTED","geometry_order":["endpoint","suffix4","suffix8"],"selected_geometry":None,"confirmation_opened":False,"complete_query_mediation_pass":False,"completed_utc":datetime.datetime.now(datetime.timezone.utc).isoformat()}; json.dump(out,open(sys.argv[1],"w"),indent=2,sort_keys=True); open(sys.argv[1],"a").write("\n")' "$ROOT/query_mediation_complete.json"
  exit 0
fi

run_trials confirmation confirmation "$CONFIRMATION_PLAN" "$selected_geometry"
DISCOVERY_GATE="$ROOT/analysis_${selected_geometry}_discovery/claim_gates.json"
CONFIRMATION_GATE="$ROOT/analysis_${selected_geometry}_confirmation/claim_gates.json"
"$PY" -c 'import datetime,json,sys; discovery=json.load(open(sys.argv[1])); confirmation=json.load(open(sys.argv[2])); passed=bool(confirmation["geometry_pass"]); out={"schema_version":"realistic_niah_v5_query_mediation_complete_v1","model_label":"Gemma4-E4B","status":"PASS" if passed else "CONFIRMATION_NEGATIVE","geometry_order":["endpoint","suffix4","suffix8"],"selected_geometry":confirmation["geometry"],"confirmation_opened":True,"discovery":discovery,"confirmation":confirmation,"complete_query_mediation_pass":passed,"complete_count_specific_mediation_pass":bool(confirmation.get("count_specific_mediation_pass",False)),"completed_utc":datetime.datetime.now(datetime.timezone.utc).isoformat()}; json.dump(out,open(sys.argv[3],"w"),indent=2,sort_keys=True); open(sys.argv[3],"a").write("\n")' "$DISCOVERY_GATE" "$CONFIRMATION_GATE" "$ROOT/query_mediation_complete.json"
