#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 3 ]]; then
  echo "usage: $0 <index|bullet> <Qwen3-8B|Gemma4-E4B> <gpu-index>" >&2
  exit 2
fi
MODE=$1
MODEL=$2
GPU_INDEX=$3

case "$MODE" in
  index) PROMPT_MODE=enumeration_index; ANCHOR_ROLE=post_marker ;;
  bullet) PROMPT_MODE=enumeration_bullet; ANCHOR_ROLE=p0_item_end ;;
  *) echo "mode must be index or bullet" >&2; exit 2 ;;
esac
case "$MODEL" in
  Qwen3-8B)
    SOURCE_LAYER=19
    LAYERS=($(seq 0 35))
    ;;
  Gemma4-E4B)
    SOURCE_LAYER=16
    LAYERS=($(seq 0 41))
    ;;
  *) echo "unsupported model: $MODEL" >&2; exit 2 ;;
esac
if [[ ! "$GPU_INDEX" =~ ^[0-9]+$ ]]; then
  echo "gpu-index must be non-negative" >&2
  exit 2
fi

ROOT=${V6_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}
PYTHON=${V6_PYTHON:-$ROOT/.venv/bin/python}
CACHE=${V6_CACHE:-$ROOT/.cache/huggingface}
RUN_BASE=${V6_RUN_BASE:-$ROOT/work/realistic_niah_v6}
REPLACEMENT_POOL=${V6_REPLACEMENT_POOL:-$RUN_BASE/replacement_seed_pool}
MODEL_ROOT=$RUN_BASE/$PROMPT_MODE/$MODEL
CONFIG=$ROOT/configs/realistic_niah_v6_${PROMPT_MODE}.json
MECHANISM=$ROOT/configs/realistic_niah_v6_${PROMPT_MODE}_count_stream_dev.json
GENERATIONS=$MODEL_ROOT/generation/generations.jsonl
COHORT_REGISTRY=$MODEL_ROOT/replacement/discovery/selected_cells.jsonl
TARGET_ROOT=$MODEL_ROOT/causal/targeted_retrieval/discovery_formal
SELECTION=$TARGET_ROOT/analysis/selection.json
SOURCE_WRITES=$TARGET_ROOT/source_writes/$ANCHOR_ROLE
COUNT_ROOT=$MODEL_ROOT/count_stream/discovery_formal
BASIS=$COUNT_ROOT/running_basis.npz
OUTPUT_ROOT=$MODEL_ROOT/causal/report_tail/discovery_formal
NATURAL_ROOT=$OUTPUT_ROOT/natural_layer_sweep
NATIVE_ROOT=$OUTPUT_ROOT/native_loop
NATIVE_COHORT_ROOT=$MODEL_ROOT/replacement/discovery_native_loop
NATIVE_COHORT_REGISTRY=$NATIVE_COHORT_ROOT/selected_cells.jsonl
REPLACEMENT_POLICY=${V6_REPLACEMENT_POLICY:-$ROOT/configs/realistic_niah_v6_replacement_policy.json}
NATIVE_POLICY=${V6_NATIVE_LOOP_POLICY:-$ROOT/configs/realistic_niah_v6_coherent_native_loop_replacement_policy.json}
RESTORE_ROOT=$OUTPUT_ROOT/restoration
LOG_ROOT=$OUTPUT_ROOT/logs

for path in "$PYTHON" "$CONFIG" "$MECHANISM" "$GENERATIONS" \
  "$COHORT_REGISTRY" "$SELECTION" "$SOURCE_WRITES/manifest.json" "$BASIS" \
  "$REPLACEMENT_POOL/stimuli.jsonl" "$REPLACEMENT_POLICY" "$NATIVE_POLICY"; do
  [[ -s "$path" ]] || { echo "missing report-tail input: $path" >&2; exit 4; }
done
SELECTED_K=$("$PYTHON" -c \
  'import json,sys; print(int(json.load(open(sys.argv[1]))["selected_k"]))' \
  "$SELECTION")
BANK_PLAN=$TARGET_ROOT/plans/k$SELECTED_K/retrieval_anchor_bank_plan.csv
[[ -s "$BANK_PLAN" ]] || { echo "missing selected bank plan: $BANK_PLAN" >&2; exit 4; }

mkdir -p "$LOG_ROOT" "$OUTPUT_ROOT/locks"
cd "$ROOT"
exec 9>"$OUTPUT_ROOT/locks/supervisor.lock"
if ! flock -n 9; then
  echo "another $MODE/$MODEL report-tail supervisor owns the lock" >&2
  exit 75
fi

run_logged() {
  local name=$1
  shift
  {
    echo "[$(date --iso-8601=seconds)] START $name"
    printf 'COMMAND env CUDA_VISIBLE_DEVICES=%q' "$GPU_INDEX"
    printf ' %q' "$@"
    printf '\n'
    env CUDA_VISIBLE_DEVICES="$GPU_INDEX" HF_HUB_OFFLINE=1 \
      TRANSFORMERS_OFFLINE=1 "$@"
    echo "[$(date --iso-8601=seconds)] PASS $name"
  } 2>&1 | tee "$LOG_ROOT/$name.log"
}

natural_kernel() {
  local name=$1
  shift
  run_logged "$name" \
    "$PYTHON" scripts/run_realistic_niah_v6_kernel.py \
    --target natural-aligned-progress --v6-config "$CONFIG" \
    --phase discovery --cohort-registry "$COHORT_REGISTRY" -- \
    --model "$MODEL" --cache-dir "$CACHE" --device-map auto \
    --torch-dtype bfloat16 --attention-backend sdpa \
    --generations "$GENERATIONS" --gold-count 10 \
    --donor-occurrence 6 --tail-offset 0 "$@"
}

natural_kernel_once() {
  local name=$1
  local output=$2
  shift 2
  if [[ -s "$output/manifest.json" ]]; then
    echo "[$(date --iso-8601=seconds)] REUSE $name verified=$output/manifest.json"
    return 0
  fi
  natural_kernel "$name" "$@" --output "$output"
}

mkdir -p "$NATURAL_ROOT/baseline" "$NATURAL_ROOT/item_end_w1" \
  "$NATURAL_ROOT/event_tail_w4" "$NATURAL_ROOT/item_span"
for direction in forward_skip backward_rewind; do
  receiver=5
  [[ "$direction" == backward_rewind ]] && receiver=7
  natural_kernel_once "natural_baseline_$direction" \
    "$NATURAL_ROOT/baseline/${direction}_k6" \
    --receiver-occurrence "$receiver" --patch-scope fixed_suffix \
    --patch-width 1 --layers 0 --conditions receiver_self native_donor
  natural_kernel_once "natural_item_end_w1_$direction" \
    "$NATURAL_ROOT/item_end_w1/${direction}_k6" \
    --receiver-occurrence "$receiver" --patch-scope fixed_suffix \
    --patch-width 1 --layers "${LAYERS[@]}" --conditions donor_to_receiver
  natural_kernel_once "natural_event_tail_w4_$direction" \
    "$NATURAL_ROOT/event_tail_w4/${direction}_k6" \
    --receiver-occurrence "$receiver" --patch-scope fixed_suffix \
    --patch-width 4 --layers "${LAYERS[@]}" --conditions donor_to_receiver
  natural_kernel_once "natural_item_span_$direction" \
    "$NATURAL_ROOT/item_span/${direction}_k6" \
    --receiver-occurrence "$receiver" --patch-scope item_span \
    --layers "${LAYERS[@]}" --conditions donor_to_receiver
done
if [[ -s "$NATURAL_ROOT/layer_sweep_analysis.json" ]]; then
  echo "[$(date --iso-8601=seconds)] REUSE natural_layer_sweep_analysis verified=$NATURAL_ROOT/layer_sweep_analysis.json"
else
  run_logged natural_layer_sweep_analysis \
    "$PYTHON" scripts/analyze_realistic_niah_v6_natural_patch_scope_layer_sweep.py \
    "$NATURAL_ROOT" --output "$NATURAL_ROOT/layer_sweep_analysis.json"
fi

if [[ -s "$NATIVE_COHORT_ROOT/native_loop_discovery.COMPLETE" && \
      -s "$NATIVE_COHORT_ROOT/manifest.json" && \
      -s "$NATIVE_COHORT_REGISTRY" ]]; then
  echo "[$(date --iso-8601=seconds)] REUSE native_loop_coherent_panel verified=$NATIVE_COHORT_ROOT/native_loop_discovery.COMPLETE"
else
  run_logged native_loop_coherent_panel \
    "$PYTHON" scripts/run_realistic_niah_v6_broad_panel_replacement.py \
    --panel-kind native_loop --v6-config "$CONFIG" \
    --mechanism-config "$MECHANISM" \
    --replacement-policy "$REPLACEMENT_POLICY" \
    --coherent-native-loop-policy "$NATIVE_POLICY" \
    --replacement-stimuli "$REPLACEMENT_POOL/stimuli.jsonl" \
    --base-cohort-registry "$COHORT_REGISTRY" --model "$MODEL" \
    --phase native_loop_discovery --generation-root "$MODEL_ROOT/generation" \
    --output "$NATIVE_COHORT_ROOT" --cache-dir "$CACHE" --device-map auto \
    --torch-dtype bfloat16 --attention-backend sdpa
fi

CONTRACT=$NATIVE_ROOT/contract
run_logged native_loop_contract \
  "$PYTHON" scripts/build_realistic_niah_v6_native_loop_contract.py \
  --model "$MODEL" --prompt-mode "$PROMPT_MODE" \
  --anchor-role "$ANCHOR_ROLE" --selection "$SELECTION" \
  --bank-plan "$BANK_PLAN" --source-writes "$SOURCE_WRITES" \
  --output "$CONTRACT"
COMPAT_SELECTION=$CONTRACT/targeted_selection_compat.json
COMPAT_ROUTING=$CONTRACT/anchor_routing_compat.json
PLAN_DIR=$NATIVE_ROOT/plan_discovery_offsets123
PLAN=$PLAN_DIR/native_loop_plan.csv

v6_count() {
  local name=$1
  shift
  run_logged "$name" \
    "$PYTHON" scripts/run_realistic_niah_v6_count_stream.py \
    --v6-config "$CONFIG" --cohort-registry "$COHORT_REGISTRY" "$@"
}

v6_count_native() {
  local name=$1
  shift
  run_logged "$name" \
    "$PYTHON" scripts/run_realistic_niah_v6_count_stream.py \
    --v6-config "$CONFIG" --cohort-registry "$NATIVE_COHORT_REGISTRY" "$@"
}

COMMON_ROWS=(
  --mechanism-config "$MECHANISM" --model "$MODEL" --cache-dir "$CACHE"
  --device-map auto --torch-dtype bfloat16 --attention-backend sdpa
  --generations "$GENERATIONS" --seed-role development --cohort one_to_one
  --donor-offsets -3 -2 -1 1 2 3 --random-seed 20260821
)
v6_count_native native_loop_plan plan-native-loop \
  "${COMMON_ROWS[@]}" --output "$PLAN_DIR"
v6_count_native native_loop_p0 p0-native-loop \
  "${COMMON_ROWS[@]}" --plan "$PLAN" --basis "$BASIS" \
  --layer "$SOURCE_LAYER" --targeted-selection "$COMPAT_SELECTION" \
  --anchor-routing "$COMPAT_ROUTING" \
  --conditions clean self_patch full_donor_patch count_subspace_transplant \
    norm_matched_orthogonal_patch count_component_removed \
    count_component_restored --max-new-tokens 48 --output "$NATIVE_ROOT/p0"
v6_count_native native_loop_boundary boundary-native-loop \
  "${COMMON_ROWS[@]}" --plan "$PLAN" --basis "$BASIS" \
  --layer "$SOURCE_LAYER" \
  --conditions clean self_patch full_donor_patch count_subspace_transplant \
    norm_matched_orthogonal_patch --max-new-tokens 64 \
  --output "$NATIVE_ROOT/boundary"
run_logged native_loop_analysis \
  "$PYTHON" scripts/analyze_realistic_niah_v6_native_loop.py \
  --v6-config "$CONFIG" --cohort-registry "$NATIVE_COHORT_REGISTRY" \
  --model "$MODEL" \
  --trials "$NATIVE_ROOT/p0" "$NATIVE_ROOT/boundary" \
  --phase discovery --output "$NATIVE_ROOT/analysis"

v6_count restoration restoration \
  --mechanism-config "$MECHANISM" --model "$MODEL" --cache-dir "$CACHE" \
  --device-map auto --torch-dtype bfloat16 --attention-backend sdpa \
  --generations "$GENERATIONS" --seed-role development --cohort one_to_one \
  --layer "$SOURCE_LAYER" \
  --conditions clean trace_token_corrupt ordinary_token_corrupt \
    trace_corrupt_full_span_restore trace_corrupt_endpoint_restore \
    trace_corrupt_marker_restore trace_corrupt_ordinary_state_patch \
    ordinary_corrupt_ordinary_state_restore --output "$RESTORE_ROOT/trials"
v6_count restoration_analysis analyze \
  --mechanism-config "$MECHANISM" --trials "$RESTORE_ROOT/trials" \
  --experiment-ids trace_source_restoration --outcome correct_count_margin \
  --strata layer --random-seed 20260820 --output "$RESTORE_ROOT/analysis"

"$PYTHON" - "$OUTPUT_ROOT" "$MODEL" "$PROMPT_MODE" "$COHORT_REGISTRY" <<'PY'
import datetime as dt
import hashlib
import json
import os
import pathlib
import sys

root = pathlib.Path(sys.argv[1])
model, prompt_mode = sys.argv[2:4]
cohort = pathlib.Path(sys.argv[4])
required = {
    "natural_analysis": root / "natural_layer_sweep/layer_sweep_analysis.json",
    "native_coherent_panel": root.parent.parent.parent / "replacement/discovery_native_loop/manifest.json",
    "native_coherent_mapping": root.parent.parent.parent / "replacement/discovery_native_loop/coherent_mapping.jsonl",
    "native_contract": root / "native_loop/contract/manifest.json",
    "native_plan": root / "native_loop/plan_discovery_offsets123/manifest.json",
    "native_p0": root / "native_loop/p0/manifest.json",
    "native_boundary": root / "native_loop/boundary/manifest.json",
    "native_analysis": root / "native_loop/analysis/claim_gates.json",
    "native_analysis_adapter": root / "native_loop/analysis/v6_analysis_manifest.json",
    "restoration": root / "restoration/trials/manifest.json",
    "restoration_analysis": root / "restoration/analysis/manifest.json",
}
missing = [str(path) for path in required.values() if not path.is_file()]
if missing:
    raise FileNotFoundError(missing)
payload = {
    "schema_version": "realistic_niah_v6_report_tail_discovery_complete_v1",
    "status": "DISCOVERY_COMPLETE",
    "model_label": model,
    "prompt_mode": prompt_mode,
    "completed_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
    "confirmation_opened": False,
    "cohort_registry": str(cohort.resolve()),
    "cohort_registry_sha256": hashlib.sha256(cohort.read_bytes()).hexdigest(),
    "panel_membership_identity": "analysis_slot_seed",
    "statistical_identity": "true_source_seed",
    "seed_aliasing": False,
    "negative_results_retained": True,
    "outputs": {
        name: {"path": str(path.resolve()), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
        for name, path in required.items()
    },
}
path = root / "report_tail_discovery_complete.json"
tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
tmp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
tmp.replace(path)
print(json.dumps(payload, sort_keys=True))
PY

printf 'PASS\n' >"$OUTPUT_ROOT/discovery.COMPLETE"
