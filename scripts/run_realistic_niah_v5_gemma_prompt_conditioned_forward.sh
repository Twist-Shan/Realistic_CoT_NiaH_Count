#!/usr/bin/env bash
set -euo pipefail

PHASE=${1:?usage: $0 discovery|confirmation GPU_INDEX}
GPU_INDEX=${2:?usage: $0 discovery|confirmation GPU_INDEX}

ROOT=/home/ubuntu/gemma_prompt_conditioned_noindex_20260827
CODE="$ROOT/code"
RESULTS="$ROOT/results"
PYTHON=/lambda/nfs/CoT-Native-thinking-v5/venv/bin/python
CACHE=/lambda/nfs/CoT-Native-thinking-v5/cache/huggingface
COHORT="$RESULTS/cohort_full_20_10/frozen_cohort.jsonl"
RUN_ROOT="$RESULTS/forward_l16_item_span/$PHASE"
SELECTION="$CODE/configs/realistic_niah_v5_gemma_shared_k6_targeted_selection_frozen.json"
ROUTING="$CODE/configs/realistic_niah_v5_gemma_shared_k6_causal_routes_frozen.json"

case "$PHASE" in
  discovery)
    SEEDS=(
      1234 1235 1236 1237 1238 1240 1245 1246 1252 1253
      1254 1255 1258 1259 1260 1264 1265 1269 1271 1274
    )
    ;;
  confirmation)
    SEEDS=(1276 1277 1278 1279 1280 1281 1282 1283 1284 1285)
    ;;
  *)
    echo "Unknown phase: $PHASE" >&2
    exit 2
    ;;
esac

mkdir -p "$RUN_ROOT"
cd "$CODE"
for DONOR in 4 6 8; do
  RECEIVER=$((DONOR - 1))
  OUTPUT="$RUN_ROOT/forward_k${DONOR}"
  env CUDA_VISIBLE_DEVICES="$GPU_INDEX" PYTHONPATH="$CODE/src" \
    "$PYTHON" scripts/run_realistic_niah_v5_natural_aligned_progress_transplant.py \
      --model Gemma4-E4B \
      --cache-dir "$CACHE" \
      --device-map auto \
      --attention-backend sdpa \
      --generations "$COHORT" \
      --cohort-mode prompt_conditioned_noindex \
      --gold-count 10 \
      --receiver-occurrence "$RECEIVER" \
      --donor-occurrence "$DONOR" \
      --tail-offset 0 \
      --patch-scope item_span \
      --layers 16 \
      --conditions receiver_self native_donor donor_to_receiver \
      --generation-conditions receiver_self donor_to_receiver \
      --max-new-tokens 96 \
      --run-attention \
      --targeted-selection "$SELECTION" \
      --targeted-routing "$ROUTING" \
      --seeds "${SEEDS[@]}" \
      --output "$OUTPUT" \
      >"$RUN_ROOT/forward_k${DONOR}.log" 2>&1
done

printf 'PASS\n' >"$RUN_ROOT/COMPLETE"
