#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 1 ]]; then
  echo "Usage: $0 RUN_ROOT" >&2
  exit 2
fi

run_root="$(readlink -f "$1")"
repo="${REALISTIC_NIAH_REPO_ROOT:-/lambda/nfs/Twist-CoT-Count-Multi-Model-v3/code/Realistic_CoT_NiaH_Count}"
python_bin="${REALISTIC_NIAH_PYTHON:-/home/ubuntu/venvs/realistic-niah-vllm/bin/python}"
state_root="${run_root}/orchestration/shard_state"

while true; do
  failed_count="$(find "${state_root}/failed" -maxdepth 1 -type f -name '*.tsv' | wc -l)"
  completed_count="$(find "${state_root}/completed" -maxdepth 1 -type f -name '*.tsv' | wc -l)"
  if [[ "${failed_count}" -gt 0 ]]; then
    echo "A V3 shard failed; finalizer is stopping." >&2
    exit 1
  fi
  if [[ "${completed_count}" -eq 48 ]]; then
    break
  fi
  if [[ "${completed_count}" -gt 48 ]]; then
    echo "Unexpected completed marker count: ${completed_count}" >&2
    exit 1
  fi
  sleep 30
done

cd "${repo}"
PYTHONPATH=src "${python_bin}" scripts/merge_realistic_niah_v3_shards.py \
  --run-root "${run_root}"
"${python_bin}" -c \
  'import json,sys; a=json.load(open(sys.argv[1],encoding="utf-8")); assert a["passed"] is True and a["requests"] == 47040 and a["unique_request_ids"] == 47040' \
  "${run_root}/orchestration/final_shard_audit.json"
