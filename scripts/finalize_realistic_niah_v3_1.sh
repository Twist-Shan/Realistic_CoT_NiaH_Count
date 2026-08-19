#!/usr/bin/env bash
set -euo pipefail

[[ $# -eq 1 ]] || { echo "Usage: $0 RUN_ROOT" >&2; exit 2; }
run_root="$(readlink -f "$1")"
repo="${REALISTIC_NIAH_REPO_ROOT:-/lambda/nfs/Twist-CoT-Count-Multi-Model-v3/code/Realistic_CoT_NiaH_Count}"
python_bin="${REALISTIC_NIAH_PYTHON:-/home/ubuntu/venvs/realistic-niah-vllm/bin/python}"
cd "${repo}"
PYTHONPATH=src "${python_bin}" scripts/audit_realistic_niah_v3_1_shard_state.py \
  --run-root "${run_root}"
PYTHONPATH=src "${python_bin}" scripts/merge_realistic_niah_v3_1_shards.py \
  --run-root "${run_root}"
"${python_bin}" -c \
  'import json,sys; a=json.load(open(sys.argv[1])); assert a["passed"] is True; assert a["requests"]==a["unique_request_ids"]==161280' \
  "${run_root}/orchestration/final_shard_audit.json"
