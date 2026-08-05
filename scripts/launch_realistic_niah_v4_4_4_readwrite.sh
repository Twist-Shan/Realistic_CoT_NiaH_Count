#!/usr/bin/env bash
set -euo pipefail

V444_RW_REPO_ROOT="${V444_RW_REPO_ROOT:-/home/ubuntu/v444_read_write/repo}"
V444_RW_PYTHON="${V444_RW_PYTHON:-/home/ubuntu/v444_read_write/.venv/bin/python}"
V444_RW_RUN_ROOT="${V444_RW_RUN_ROOT:-/lambda/nfs/CoT-Non-thinking-v4/runs/v4_4_4_natural_ov/run_20260803_v4_4_4_natural_ov_qwen_l28_a100_1501368870_v1}"
V444_RW_LOG_PATH="${V444_RW_LOG_PATH:-/home/ubuntu/v444_read_write/launcher.log}"
V444_RW_EXIT_PATH="${V444_RW_EXIT_PATH:-/home/ubuntu/v444_read_write/launcher.exit}"

cd "${V444_RW_REPO_ROOT}"
export PYTHONPATH="${V444_RW_REPO_ROOT}/src"

set +e
"${V444_RW_PYTHON}" scripts/run_realistic_niah_v4_4_4_readwrite.py \
  --stage campaign \
  --resume \
  --run-root "${V444_RW_RUN_ROOT}" \
  --repo-root "${V444_RW_REPO_ROOT}" \
  2>&1 | tee "${V444_RW_LOG_PATH}"
V444_RW_STATUS=${PIPESTATUS[0]}
set -e
printf '%s\n' "${V444_RW_STATUS}" > "${V444_RW_EXIT_PATH}"
exit "${V444_RW_STATUS}"
