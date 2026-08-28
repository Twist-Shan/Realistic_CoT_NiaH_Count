#!/usr/bin/env bash
set -euo pipefail

if [[ "$#" -lt 5 ]]; then
  printf 'usage: %s GPU INPUT LABEL SEED [SEED ...]\n' "$0" >&2
  exit 2
fi

GPU=$1
INPUT=$2
LABEL=$3
shift 3
SEEDS=("$@")

ROOT=/home/ubuntu/same_site_progress_20260827
PYTHON=/lambda/nfs/CoT-Native-thinking-v5/venv/bin/python
CACHE=/lambda/nfs/CoT-Native-thinking-v5/cache/huggingface
OUTPUT="$ROOT/work/output/n10_natural_crossk_attention_generation_v1/$LABEL"
mkdir -p "$OUTPUT"
cd "$ROOT"

for DONOR in 4 6 8; do
  for DIRECTION in forward_skip backward_rewind; do
    if [[ "$DIRECTION" == forward_skip ]]; then
      RECEIVER=$((DONOR - 1))
    else
      RECEIVER=$((DONOR + 1))
    fi
    CUDA_VISIBLE_DEVICES="$GPU" "$PYTHON" \
      scripts/run_realistic_niah_v5_natural_aligned_progress_transplant.py \
      --model Qwen3-8B \
      --cache-dir "$CACHE" \
      --device-map auto \
      --generations "$INPUT" \
      --gold-count 10 \
      --receiver-occurrence "$RECEIVER" \
      --donor-occurrence "$DONOR" \
      --tail-offset 0 \
      --patch-width 4 \
      --layers 16 \
      --conditions receiver_self native_donor donor_to_receiver \
      --generation-conditions receiver_self donor_to_receiver \
      --max-new-tokens 96 \
      --run-attention \
      --targeted-selection configs/realistic_niah_v5_qwen_shared_k128_targeted_selection_frozen.json \
      --targeted-routing configs/realistic_niah_v5_qwen_shared_k128_causal_routes_frozen.json \
      --seeds "${SEEDS[@]}" \
      --output "$OUTPUT/${DIRECTION}_k${DONOR}" \
      >"$OUTPUT/${DIRECTION}_k${DONOR}.log" 2>&1
  done
done
