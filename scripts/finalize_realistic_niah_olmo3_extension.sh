#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 1 ]]; then
  echo "Usage: $0 RUN_ROOT" >&2
  exit 2
fi

run_root="$(readlink -f "$1")"
repo="${REALISTIC_NIAH_REPO_ROOT:-/lambda/nfs/Twist-CoT-Count-Multi-Model-v2/code/Realistic_CoT_NiaH_Count}"
python_bin="${REALISTIC_NIAH_PYTHON:-/home/ubuntu/venvs/realistic-niah-vllm/bin/python}"
state_root="${run_root}/orchestration/shard_state"
expected_shards=4

case "${run_root}" in
  */runs/realistic_niah_v2/olmo3_7b_extension_*)
    ;;
  *)
    echo "Refusing unexpected OLMo extension run root: ${run_root}" >&2
    exit 2
    ;;
esac

mkdir -p "${state_root}/completed" "${state_root}/failed"

while true; do
  mapfile -t failed < <(
    find "${state_root}/failed" -maxdepth 1 -type f -name '*.tsv' -size +0c \
      2>/dev/null | sort
  )
  if [[ "${#failed[@]}" -gt 0 ]]; then
    echo "OLMo extension has failed shard markers:" >&2
    printf '%s\n' "${failed[@]}" >&2
    exit 1
  fi
  completed="$(
    find "${state_root}/completed" -maxdepth 1 -type f -name '*.tsv' -size +0c \
      2>/dev/null | wc -l
  )"
  if [[ "${completed}" -eq "${expected_shards}" ]]; then
    break
  fi
  if [[ "${completed}" -gt "${expected_shards}" ]]; then
    echo "Found more completion markers than registered shards" >&2
    exit 1
  fi
  printf '%s OLMo extension: %s/%s shards complete\n' \
    "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "${completed}" "${expected_shards}"
  sleep 15
done

cd "${repo}"
PYTHONPATH=src "${python_bin}" \
  scripts/merge_realistic_niah_olmo3_extension_shards.py \
  --run-root "${run_root}"
"${python_bin}" -c \
  'import json,sys; a=json.load(open(sys.argv[1],encoding="utf-8")); assert a["passed"] is True; assert a["requests"] == a["unique_request_ids"] == 2000' \
  "${run_root}/orchestration/final_shard_audit.json"
echo "OLMo 3 extension final audit passed: ${run_root}"
