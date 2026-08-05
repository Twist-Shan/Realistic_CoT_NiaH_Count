#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/home/ubuntu/v444_upstream_path/repo}"
RUN_ROOT="${RUN_ROOT:-/lambda/nfs/CoT-Non-thinking-v4/runs/v4_4_4_natural_ov/run_20260804_v4_4_4_upstream_confirmation_qwen_l28_a100_1501366726_v1}"
CACHE_DIR="${CACHE_DIR:-/lambda/nfs/CoT-Non-thinking-v4/hf-cache}"
PYTHON_BIN="${PYTHON_BIN:-/home/ubuntu/v444_read_write/.venv/bin/python}"
LOG_FILE="${LOG_FILE:-/home/ubuntu/v444_upstream_path/upstream_confirmation_full.log}"

exec > >(tee -a "$LOG_FILE") 2>&1

cd "$REPO_ROOT"
export PYTHONPATH="$REPO_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
export TOKENIZERS_PARALLELISM=false

"$PYTHON_BIN" scripts/run_realistic_niah_v4_4_4.py \
  --stage dataset \
  --repo-root "$REPO_ROOT" \
  --run-root "$RUN_ROOT" \
  --config configs/realistic_niah_v4_4_4_upstream_confirmation_base.json \
  --cache-dir "$CACHE_DIR" \
  --resume

exec "$PYTHON_BIN" scripts/run_realistic_niah_v4_4_4_upstream_confirmation.py \
  --stage campaign \
  --repo-root "$REPO_ROOT" \
  --run-root "$RUN_ROOT" \
  --config configs/realistic_niah_v4_4_4_upstream_confirmation.json \
  --cache-dir "$CACHE_DIR" \
  --device-map auto \
  --resume
