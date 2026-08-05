#!/usr/bin/env bash
set -euo pipefail

V444_RELAY_REPO_ROOT="${V444_RELAY_REPO_ROOT:-/home/ubuntu/v444_natural_ov/repo}"
V444_RELAY_PYTHON="${V444_RELAY_PYTHON:-/home/ubuntu/v443_ov_causal/.venv/bin/python}"
V444_RELAY_RUN_ROOT="${V444_RELAY_RUN_ROOT:-/lambda/nfs/CoT-Non-thinking-v4/runs/v4_4_4_natural_ov/run_20260803_v4_4_4_natural_ov_qwen_l28_a100_1501368870_v1}"
V444_RELAY_LOG_PATH="${V444_RELAY_LOG_PATH:-/home/ubuntu/v444_natural_ov/relay_launcher.log}"
V444_RELAY_EXIT_PATH="${V444_RELAY_EXIT_PATH:-/home/ubuntu/v444_natural_ov/relay_launcher.exit}"

cd "${V444_RELAY_REPO_ROOT}"
export PYTHONPATH="${V444_RELAY_REPO_ROOT}/src"

set +e
"${V444_RELAY_PYTHON}" scripts/run_realistic_niah_v4_4_4_relay.py \
  --stage campaign \
  --resume \
  --run-root "${V444_RELAY_RUN_ROOT}" \
  --repo-root "${V444_RELAY_REPO_ROOT}" \
  2>&1 | tee "${V444_RELAY_LOG_PATH}"
V444_RELAY_STATUS=${PIPESTATUS[0]}
set -e
printf '%s\n' "${V444_RELAY_STATUS}" > "${V444_RELAY_EXIT_PATH}"
exit "${V444_RELAY_STATUS}"
