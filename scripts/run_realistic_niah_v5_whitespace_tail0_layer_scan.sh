#!/usr/bin/env bash
set -euo pipefail

ROOT=/home/ubuntu/same_site_progress_20260827
PYTHON=/lambda/nfs/CoT-Native-thinking-v5/venv/bin/python
OUTPUT="$ROOT/work/output/n10_natural_site_layer_localization_v1"
cd "$ROOT"
CUDA_VISIBLE_DEVICES=0 "$PYTHON" \
  scripts/run_realistic_niah_v5_natural_aligned_progress_transplant.py \
  --model Qwen3-8B \
  --cache-dir /lambda/nfs/CoT-Native-thinking-v5/cache/huggingface \
  --device-map auto \
  --generations "$ROOT/work/input/n10/discovery_rows_first_pass_noindex_v5.jsonl" \
  --gold-count 10 \
  --receiver-occurrence 5 \
  --donor-occurrence 6 \
  --tail-offset 0 \
  --patch-width 1 \
  --layers 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 32 33 34 35 \
  --conditions receiver_self native_donor donor_to_receiver \
  --seeds 1293 1359 1384 1539 2322 \
  --output "$OUTPUT/whitespace_tail0_k6_v2"
