#!/usr/bin/env bash
set -euo pipefail

MODEL=${1:?usage: $0 MODEL GPU_INDEX}
GPU_INDEX=${2:?usage: $0 MODEL GPU_INDEX}
ROOT_DIR=${ROOT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}
cd "$ROOT_DIR"

PYTHON=${PYTHON:-$ROOT_DIR/.venv/bin/python}
CACHE_DIR=${CACHE_DIR:-$ROOT_DIR/work/hf_cache}
ALIGNED="$ROOT_DIR/work/v5_native_sample_aligned_20260829"
RUN_ROOT="$ALIGNED/runs/$MODEL"
INPUT_ROOT="$ALIGNED/inputs/$MODEL"
LOG="$RUN_ROOT/logs/supervisor.log"
LOCK="$RUN_ROOT/locks/supervisor.lock"
SUITE_COMPLETE="$RUN_ROOT/suite_complete.json"

V5_CONFIG="$ROOT_DIR/configs/realistic_niah_v5.json"
MECH_DEV="$ROOT_DIR/configs/realistic_niah_v5_native_count_stream_dev.json"
MECH_CONFIRM="$ROOT_DIR/configs/realistic_niah_v5_native_count_stream_confirmation_v1.json"
GENERATIONS="$ROOT_DIR/work/v5_trace_parser_v2/${MODEL}_generations_reparsed.jsonl"
ALIGNED_GENERATIONS="$ALIGNED/generation_views_routed/${MODEL}_generations_aligned.jsonl"
ROUTED_PANEL="$ALIGNED/shared_routed_transition_panel/${MODEL}_anchor_panel.jsonl"
GRAMMAR_PANEL="$ALIGNED/shared_grammar_panel_v2/${MODEL}_anchor_panel.jsonl"
GRAMMAR_MANIFEST="$ALIGNED/shared_grammar_panel_v2/${MODEL}_manifest.json"
AQ_PAIRS="$ALIGNED/answer_query_layer_sweep_plan/${MODEL}_pairs.jsonl"
INDEXED_PANEL="$ALIGNED/indexed_progress_control_panel/${MODEL}.jsonl"
INDEXED_FREEZE="$ALIGNED/indexed_progress_control_freeze.json"
BANK_PLAN="$INPUT_ROOT/frozen_targeted_count_plan.csv"
BASIS="$INPUT_ROOT/item_end_discovery_basis.npz"
SELECTION=""
ROUTING=""
SOURCE_LAYER=""
AQ_LAYERS=()
INDEXED_LAYERS=()
INDEXED_LAYER=""
DISCOVERY_SEEDS=(1234 1235 1236 1237 1238 1239 1240 1241 1242 1243 1244 1245 1246 1247 1248 1249 1250 1251 1252 1253)
CONFIRMATION_SEEDS=(1254 1255 1256 1257 1258 1259 1260 1261 1262 1263)

case "$MODEL" in
  Qwen3-8B)
    SOURCE_LAYER=19
    SELECTION="$ROOT_DIR/configs/realistic_niah_v5_qwen_shared_k128_targeted_selection_frozen.json"
    ROUTING="$ROOT_DIR/configs/realistic_niah_v5_qwen_shared_k128_causal_routes_frozen.json"
    AQ_LAYERS=(0 5 10 15 20 25 30 35)
    INDEXED_LAYERS=($(seq 0 35))
    INDEXED_LAYER=19
    ;;
  Gemma4-E4B)
    SOURCE_LAYER=16
    SELECTION="$ROOT_DIR/configs/realistic_niah_v5_gemma_shared_k6_targeted_selection_frozen.json"
    ROUTING="$ROOT_DIR/configs/realistic_niah_v5_gemma_shared_k6_causal_routes_frozen.json"
    AQ_LAYERS=(0 6 12 18 23 29 35 41)
    INDEXED_LAYERS=($(seq 0 41))
    INDEXED_LAYER=16
    ;;
  *)
    echo "unsupported model: $MODEL" >&2
    exit 2
    ;;
esac

mkdir -p "$RUN_ROOT/logs" "$RUN_ROOT/locks" "$CACHE_DIR"
exec 9>"$LOCK"
if ! flock -n 9; then
  echo "another aligned-suite supervisor owns $MODEL" >&2
  exit 75
fi

for path in \
  "$PYTHON" "$V5_CONFIG" "$MECH_DEV" "$MECH_CONFIRM" \
  "$GENERATIONS" "$ALIGNED_GENERATIONS" "$ROUTED_PANEL" \
  "$GRAMMAR_PANEL" "$GRAMMAR_MANIFEST" "$AQ_PAIRS" \
  "$INDEXED_PANEL" "$INDEXED_FREEZE" "$BANK_PLAN" \
  "$BASIS" "$SELECTION" "$ROUTING" "$ALIGNED/alignment_audit.json"; do
  test -s "$path" || { echo "missing required aligned-suite input: $path" >&2; exit 4; }
done

"$PYTHON" - "$ALIGNED/alignment_audit.json" "$INDEXED_FREEZE" "$MODEL" "$INDEXED_LAYER" <<'PY'
import json, pathlib, sys
value = json.loads(pathlib.Path(sys.argv[1]).read_text(encoding="utf-8"))
assert value["status"] == "PASS"
assert all(item["status"] == "PASS" for item in value["evidence"].values())
freeze = json.loads(pathlib.Path(sys.argv[2]).read_text(encoding="utf-8"))
assert freeze["status"] == "FROZEN_BEFORE_CONFIRMATION"
assert freeze["confirmation_outcomes_observed"] is False
assert int(freeze["active_confirmation_layers"][sys.argv[3]]) == int(sys.argv[4])
PY

run_logged() {
  local label=$1
  shift
  echo "START model=$MODEL stage=$label utc=$(date -u +%FT%TZ)" | tee -a "$LOG"
  "$@" 2>&1 | tee -a "$LOG"
  echo "COMPLETE model=$MODEL stage=$label utc=$(date -u +%FT%TZ)" | tee -a "$LOG"
}

run_if_missing() {
  local label=$1
  local sentinel=$2
  shift 2
  if [[ -s "$sentinel" ]]; then
    echo "REUSE model=$MODEL stage=$label sentinel=$sentinel" | tee -a "$LOG"
    return 0
  fi
  run_logged "$label" "$@"
  test -s "$sentinel"
}

model_python() {
  CUDA_VISIBLE_DEVICES="$GPU_INDEX" "$PYTHON" "$@"
}

# 1. Targeted retrieval on the exact 20+10 shared routed-transition panel.
TARGET_ROOT="$RUN_ROOT/targeted_retrieval"
for spec in "development discovery" "confirmation confirmation"; do
  read -r role phase <<<"$spec"
  trials="$TARGET_ROOT/$phase"
  analysis="$TARGET_ROOT/analysis_$phase"
  run_if_missing "targeted_retrieval_$phase" "$trials/manifest.json" \
    model_python "$ROOT_DIR/scripts/run_realistic_niah_v5.py" causal-heads-behavior \
      --config "$V5_CONFIG" --model "$MODEL" --cache-dir "$CACHE_DIR" \
      --device-map auto --torch-dtype bfloat16 --attention-backend sdpa \
      --generations "$GENERATIONS" --plan "$BANK_PLAN" \
      --anchor-routing "$ROUTING" --behavior-all-routed-grammars \
      --allow-selection-scope-bank-transfer \
      --conditions clean selected_bank layer_matched_random --include-secondary \
      --counts 2 3 4 5 6 7 8 9 10 --limit 300 --max-new-tokens 512 \
      --decode-head-ablation-steps -1 --evaluation-split "$phase" \
      --anchor-registry-input "$ROUTED_PANEL" --output "$trials"
  run_if_missing "targeted_retrieval_analyze_$phase" "$analysis/claim_gates.json" \
    "$PYTHON" "$ROOT_DIR/scripts/analyze_realistic_niah_v5_targeted_count_endpoint.py" \
      --trials "$trials" --output "$analysis" --phase "$phase"
done

# 2. Retrieval-query -> carrier -> commit write mediation, same routed rows.
WRITE_ROOT="$RUN_ROOT/targeted_counter_write"
for spec in "development discovery $MECH_DEV" "confirmation confirmation $MECH_CONFIRM"; do
  read -r role phase mechanism <<<"$spec"
  trials="$WRITE_ROOT/$phase"
  analysis="$WRITE_ROOT/analysis_$phase"
  run_if_missing "targeted_counter_write_$phase" "$trials/manifest.json" \
    model_python "$ROOT_DIR/scripts/run_realistic_niah_v5_targeted_counter_write.py" \
      --mechanism-config "$mechanism" --v5-config "$V5_CONFIG" \
      --model "$MODEL" --cache-dir "$CACHE_DIR" --device-map auto \
      --torch-dtype bfloat16 --attention-backend sdpa \
      --generations "$GENERATIONS" --seed-role "$role" \
      --anchor-registry "$ROUTED_PANEL" --targeted-registry "$ROUTED_PANEL" \
      --bank-plan "$BANK_PLAN" --source-layer "$SOURCE_LAYER" --resume \
      --output "$trials"
  run_if_missing "targeted_counter_write_analyze_$phase" "$analysis/claim_gates.json" \
    "$PYTHON" "$ROOT_DIR/scripts/analyze_realistic_niah_v5_targeted_counter_write.py" \
      --input "$trials" --phase "$phase" --random-seed 20260829 \
      --output "$analysis"
done

# 3. Same-timing grammar span panel (real common support: 19+9).
GRAMMAR_ROOT="$RUN_ROOT/grammar_span"
for spec in "development discovery $MECH_DEV" "confirmation confirmation $MECH_CONFIRM"; do
  read -r role phase mechanism <<<"$spec"
  trials="$GRAMMAR_ROOT/$phase"
  analysis="$GRAMMAR_ROOT/analysis_$phase"
  run_if_missing "grammar_span_$phase" "$trials/manifest.json" \
    model_python "$ROOT_DIR/scripts/run_realistic_niah_v5_grammar_span_decomposition.py" \
      --mechanism-config "$mechanism" --v5-config "$V5_CONFIG" \
      --model "$MODEL" --cache-dir "$CACHE_DIR" --device-map auto \
      --torch-dtype bfloat16 --attention-backend sdpa \
      --generations "$GENERATIONS" --seed-role "$role" \
      --anchor-panel "$GRAMMAR_PANEL" --anchor-manifest "$GRAMMAR_MANIFEST" \
      --layer "$SOURCE_LAYER" --max-new-tokens 16 --resume --output "$trials"
  run_if_missing "grammar_span_analyze_$phase" "$analysis/claim_gates.json" \
    "$PYTHON" "$ROOT_DIR/scripts/analyze_realistic_niah_v5_grammar_span_decomposition.py" \
      --input "$trials" --phase "$phase" --bootstrap-samples 10000 \
      --random-seed 20260829 --output "$analysis"
done

# 4. Timing-stratified NCC and direct next-count logit margin on identical cells.
NCC_ROOT="$RUN_ROOT/stratified_ncc"
MARGIN_ROOT="$RUN_ROOT/direct_count_logit_margin"
for timing in rank_after_city rank_before_city; do
  panel="$ALIGNED/shared_grammar_panel_v2/${MODEL}_${timing}_panel.jsonl"
  timing_bank="$INPUT_ROOT/${timing}_bank_plan.csv"
  test -s "$panel" && test -s "$timing_bank"
  for spec in "development discovery $MECH_DEV" "confirmation confirmation $MECH_CONFIRM"; do
    read -r role phase mechanism <<<"$spec"
    ncc_trials="$NCC_ROOT/$timing/$phase"
    margin_trials="$MARGIN_ROOT/$timing/$phase"
    run_if_missing "ncc_${timing}_$phase" "$ncc_trials/manifest.json" \
      model_python "$ROOT_DIR/scripts/run_realistic_niah_v5_stratified_targeted_counter_ncc.py" \
        --mechanism-config "$mechanism" --v5-config "$V5_CONFIG" \
        --model "$MODEL" --cache-dir "$CACHE_DIR" --device-map auto \
        --torch-dtype bfloat16 --attention-backend sdpa \
        --generations "$GENERATIONS" --seed-role "$role" --timing "$timing" \
        --panel "$panel" --bank-plan "$timing_bank" --resume --output "$ncc_trials"
    run_if_missing "margin_${timing}_$phase" "$margin_trials/manifest.json" \
      model_python "$ROOT_DIR/scripts/run_realistic_niah_v5_targeted_counter_logit_margin.py" \
        --mechanism-config "$mechanism" --v5-config "$V5_CONFIG" \
        --model "$MODEL" --cache-dir "$CACHE_DIR" --device-map auto \
        --torch-dtype bfloat16 --attention-backend sdpa \
        --generations "$GENERATIONS" --seed-role "$role" --timing "$timing" \
        --panel "$panel" --bank-plan "$timing_bank" --resume --output "$margin_trials"
  done
  run_if_missing "ncc_analyze_$timing" "$NCC_ROOT/$timing/analysis/claim_gates.json" \
    "$PYTHON" "$ROOT_DIR/scripts/analyze_realistic_niah_v5_stratified_targeted_counter_ncc.py" \
      --discovery "$NCC_ROOT/$timing/discovery" \
      --confirmation "$NCC_ROOT/$timing/confirmation" --timing "$timing" \
      --output "$NCC_ROOT/$timing/analysis"
  run_if_missing "margin_analyze_$timing" "$MARGIN_ROOT/$timing/analysis/claim_gates.json" \
    "$PYTHON" "$ROOT_DIR/scripts/analyze_realistic_niah_v5_targeted_counter_logit_margin.py" \
      --discovery "$MARGIN_ROOT/$timing/discovery" \
      --confirmation "$MARGIN_ROOT/$timing/confirmation" --timing "$timing" \
      --output "$MARGIN_ROOT/$timing/analysis"
done
run_if_missing "ncc_finalize" "$NCC_ROOT/stratified_ncc_complete.json" \
  "$PYTHON" "$ROOT_DIR/scripts/finalize_realistic_niah_v5_stratified_targeted_counter_ncc.py" \
    --model "$MODEL" --output-root "$NCC_ROOT" \
    --output "$NCC_ROOT/stratified_ncc_complete.json"

# 5. Position-preserving next-city token blanks on the same 30 transitions.
TOKEN_ROOT="$RUN_ROOT/next_city_token_blank"
run_if_missing "next_city_token_blank" "$TOKEN_ROOT/worker_00_manifest.json" \
  model_python "$ROOT_DIR/scripts/run_realistic_niah_v5_token_level_ablation.py" \
    --mode targeting --config "$V5_CONFIG" --generations "$GENERATIONS" \
    --output "$TOKEN_ROOT" --model "$MODEL" --cache-dir "$CACHE_DIR" \
    --device-map auto --torch-dtype bfloat16 --attention-backend sdpa \
    --bank-plan "$BANK_PLAN" --anchor-registry "$ROUTED_PANEL" --split all \
    --registry-anchor-match exact \
    --matched-control-repeats 3 --max-new-tokens 32
run_if_missing "next_city_token_blank_analyze" "$TOKEN_ROOT/analysis/analysis_audit.json" \
  "$PYTHON" "$ROOT_DIR/scripts/analyze_realistic_niah_v5_token_level_ablation.py" \
    --input "$TOKEN_ROOT" --output "$TOKEN_ROOT/analysis"

# 6. Full answer-query state patching on the exact same 40 directed pairs.
AQ_ROOT="$RUN_ROOT/answer_query_layer_sweep"
run_if_missing "answer_query_layer_sweep" "$AQ_ROOT/trials.jsonl" \
  model_python "$ROOT_DIR/scripts/run_realistic_niah_v5.py" causal-patch \
    --config "$V5_CONFIG" --model "$MODEL" --cache-dir "$CACHE_DIR" \
    --device-map auto --torch-dtype bfloat16 --attention-backend sdpa \
    --generations "$GENERATIONS" --pairs "$AQ_PAIRS" \
    --output "$AQ_ROOT/trials.jsonl" --layers "${AQ_LAYERS[@]}" \
    --conditions self_patch full_donor_patch \
    --receiver-site-id answer_query_v3 --donor-site-id answer_query_v3 \
    --max-new-tokens 16 --restartable
run_if_missing "answer_query_layer_sweep_analyze" "$AQ_ROOT/analysis/audit.json" \
  "$PYTHON" "$ROOT_DIR/scripts/analyze_v5_answer_query_layer_sweep.py" \
    --trials "$AQ_ROOT/trials.jsonl" --pairs "$AQ_PAIRS" \
    --output-dir "$AQ_ROOT/analysis" --expected-layers "${AQ_LAYERS[@]}"

# 7. Full commit/event state -> next targeted query, exact +/-1 structural pairs.
COMMIT_ROOT="$RUN_ROOT/commit_state_to_targeted_query"
for spec in "development discovery $MECH_DEV" "confirmation confirmation $MECH_CONFIRM"; do
  read -r role phase mechanism <<<"$spec"
  plan="$ALIGNED/native_loop_plans/$MODEL/$phase/native_loop_plan.csv"
  trials="$COMMIT_ROOT/$phase"
  analysis="$COMMIT_ROOT/analysis_$phase"
  test -s "$plan"
  run_if_missing "commit_to_query_$phase" "$trials/manifest.json" \
    model_python "$ROOT_DIR/scripts/run_realistic_niah_v5_count_stream.py" p0-native-loop \
      --mechanism-config "$mechanism" --v5-config "$V5_CONFIG" \
      --model "$MODEL" --cache-dir "$CACHE_DIR" --device-map auto \
      --torch-dtype bfloat16 --attention-backend sdpa \
      --generations "$ALIGNED_GENERATIONS" --seed-role "$role" \
      --cohort parser_hit --plan "$plan" --basis "$BASIS" \
      --layer "$SOURCE_LAYER" --targeted-selection "$SELECTION" \
      --anchor-routing "$ROUTING" \
      --conditions clean self_patch full_donor_patch count_subspace_transplant \
        norm_matched_orthogonal_patch \
      --donor-offsets -1 1 --random-seed 20260829 \
      --allow-incomplete-offsets --no-boundaries --skip-greedy --output "$trials"
  run_if_missing "commit_to_query_analyze_$phase" "$analysis/claim_gates.json" \
    "$PYTHON" "$ROOT_DIR/scripts/analyze_realistic_niah_v5_commit_state_to_targeted_query.py" \
      --trials "$trials" --phase "$phase" --bootstrap-samples 10000 \
      --random-seed 20260829 --output "$analysis"
done

# 8. Surface-matched explicit-index positive control.  Both models use the
# exact same 20+10 seeds, N=10 records, numbered-list body, k grid, directions,
# conditions, and generation budget.  Only native channel wrappers and the
# architecture-specific layer axes differ.  Confirmation layers were frozen
# from the aligned running-representation discovery analysis, not from these
# indexed outcomes.
INDEXED_ROOT="$RUN_ROOT/indexed_progress_control"
INDEXED_DISCOVERY="$INDEXED_ROOT/discovery_layer_sweep"
for direction in forward_skip backward_rewind; do
  receiver=5
  if [[ "$direction" == "backward_rewind" ]]; then
    receiver=7
  fi
  baseline="$INDEXED_DISCOVERY/baseline/${direction}_k6"
  item_span="$INDEXED_DISCOVERY/item_span/${direction}_k6"
  run_if_missing "indexed_discovery_baseline_${direction}" "$baseline/trials.jsonl" \
    model_python "$ROOT_DIR/scripts/run_realistic_niah_v5_natural_aligned_progress_transplant.py" \
      --model "$MODEL" --cache-dir "$CACHE_DIR" --device-map auto \
      --torch-dtype bfloat16 --attention-backend sdpa \
      --generations "$INDEXED_PANEL" --cohort-mode indexed_positive_control \
      --gold-count 10 --receiver-occurrence "$receiver" --donor-occurrence 6 \
      --tail-offset 0 --patch-scope fixed_suffix --patch-width 1 \
      --layers 0 --conditions receiver_self native_donor \
      --seeds "${DISCOVERY_SEEDS[@]}" --output "$baseline"
  run_if_missing "indexed_discovery_item_span_${direction}" "$item_span/trials.jsonl" \
    model_python "$ROOT_DIR/scripts/run_realistic_niah_v5_natural_aligned_progress_transplant.py" \
      --model "$MODEL" --cache-dir "$CACHE_DIR" --device-map auto \
      --torch-dtype bfloat16 --attention-backend sdpa \
      --generations "$INDEXED_PANEL" --cohort-mode indexed_positive_control \
      --gold-count 10 --receiver-occurrence "$receiver" --donor-occurrence 6 \
      --tail-offset 0 --patch-scope item_span \
      --layers "${INDEXED_LAYERS[@]}" --conditions donor_to_receiver \
      --seeds "${DISCOVERY_SEEDS[@]}" --output "$item_span"
done
run_if_missing "indexed_discovery_analyze" "$INDEXED_DISCOVERY/layer_sweep_analysis.json" \
  "$PYTHON" "$ROOT_DIR/scripts/analyze_realistic_niah_v5_natural_patch_scope_layer_sweep.py" \
    "$INDEXED_DISCOVERY" --output "$INDEXED_DISCOVERY/layer_sweep_analysis.json"

INDEXED_CONFIRMATION_RUNS="$INDEXED_ROOT/confirmation_runs/confirmation/item_span"
for donor in 4 6 8; do
  for direction in forward_skip backward_rewind; do
    receiver=$((donor - 1))
    if [[ "$direction" == "backward_rewind" ]]; then
      receiver=$((donor + 1))
    fi
    output="$INDEXED_CONFIRMATION_RUNS/${direction}_k${donor}"
    run_if_missing "indexed_confirmation_${direction}_k${donor}" "$output/trials.jsonl" \
      model_python "$ROOT_DIR/scripts/run_realistic_niah_v5_natural_aligned_progress_transplant.py" \
        --model "$MODEL" --cache-dir "$CACHE_DIR" --device-map auto \
        --torch-dtype bfloat16 --attention-backend sdpa \
        --generations "$INDEXED_PANEL" --cohort-mode indexed_positive_control \
        --gold-count 10 --receiver-occurrence "$receiver" --donor-occurrence "$donor" \
        --tail-offset 0 --patch-scope item_span --layers "$INDEXED_LAYER" \
        --conditions receiver_self native_donor donor_to_receiver \
        --generation-conditions receiver_self donor_to_receiver \
        --max-new-tokens 96 --run-attention \
        --targeted-selection "$SELECTION" --targeted-routing "$ROUTING" \
        --seeds "${CONFIRMATION_SEEDS[@]}" --output "$output"
  done
done
run_if_missing "indexed_confirmation_analyze" "$INDEXED_ROOT/confirmation_analysis.json" \
  "$PYTHON" "$ROOT_DIR/scripts/analyze_realistic_niah_v5_natural_patch_scope_frozen.py" \
    "$INDEXED_ROOT/confirmation_runs" --output "$INDEXED_ROOT/confirmation_analysis.json"

"$PYTHON" - "$MODEL" "$RUN_ROOT" "$SUITE_COMPLETE" <<'PY'
import datetime as dt, hashlib, json, pathlib, sys
model, root_raw, output_raw = sys.argv[1:]
root = pathlib.Path(root_raw)
required = [
    root / "targeted_retrieval/analysis_discovery/claim_gates.json",
    root / "targeted_retrieval/analysis_confirmation/claim_gates.json",
    root / "targeted_counter_write/analysis_discovery/claim_gates.json",
    root / "targeted_counter_write/analysis_confirmation/claim_gates.json",
    root / "grammar_span/analysis_discovery/claim_gates.json",
    root / "grammar_span/analysis_confirmation/claim_gates.json",
    root / "stratified_ncc/stratified_ncc_complete.json",
    root / "next_city_token_blank/analysis/analysis_audit.json",
    root / "answer_query_layer_sweep/analysis/audit.json",
    root / "commit_state_to_targeted_query/analysis_discovery/claim_gates.json",
    root / "commit_state_to_targeted_query/analysis_confirmation/claim_gates.json",
    root / "indexed_progress_control/discovery_layer_sweep/layer_sweep_analysis.json",
    root / "indexed_progress_control/confirmation_analysis.json",
]
missing = [str(path) for path in required if not path.is_file() or not path.stat().st_size]
if missing:
    raise FileNotFoundError(missing)
value = {
    "schema_version": "realistic_niah_v5_cross_model_sample_aligned_suite_v1",
    "status": "PASS",
    "model_label": model,
    "completed_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
    "exact_cross_model_sample_alignment_required": True,
    "evidence_sha256": {
        str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in required
    },
}
pathlib.Path(output_raw).write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
print(json.dumps({"status": "PASS", "model": model}, sort_keys=True))
PY

echo "ALIGNED_SUITE_COMPLETE model=$MODEL utc=$(date -u +%FT%TZ)" | tee -a "$LOG"
