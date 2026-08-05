#!/usr/bin/env bash
set -euo pipefail

V444_REPO_ROOT="${V444_REPO_ROOT:-/home/ubuntu/v444_natural_ov/repo}"
V444_PYTHON="${V444_PYTHON:-/home/ubuntu/v443_ov_causal/.venv/bin/python}"
V444_RUN_ROOT="${V444_RUN_ROOT:-/lambda/nfs/CoT-Non-thinking-v4/runs/v4_4_4_natural_ov/run_20260803_v4_4_4_natural_ov_qwen_l28_a100_1501368870_v1}"
V444_SOURCE_ROOT="${V444_SOURCE_ROOT:-/lambda/nfs/CoT-Non-thinking-v4/runs/run_20260731_v4_numeric_presentation_v3}"
V444_NAMESPACE_ROOT="${V444_NAMESPACE_ROOT:-/lambda/nfs/CoT-Non-thinking-v4/runs/v4_4_4_natural_ov}"
V444_LOG_PATH="${V444_LOG_PATH:-/home/ubuntu/v444_natural_ov/full_launcher.log}"
V444_EXIT_PATH="${V444_EXIT_PATH:-/home/ubuntu/v444_natural_ov/full_launcher.exit}"

cd "${V444_REPO_ROOT}"
export PYTHONPATH="${V444_REPO_ROOT}/src"

set +e
"${V444_PYTHON}" scripts/run_realistic_niah_v4_4_4.py \
  --stage campaign \
  --resume \
  --run-root "${V444_RUN_ROOT}" \
  --source-run-root "${V444_SOURCE_ROOT}" \
  --output-namespace-root "${V444_NAMESPACE_ROOT}" \
  --repo-root "${V444_REPO_ROOT}" \
  2>&1 | tee "${V444_LOG_PATH}"
V444_STATUS=${PIPESTATUS[0]}
set -e
printf '%s\n' "${V444_STATUS}" > "${V444_EXIT_PATH}"
exit "${V444_STATUS}"

