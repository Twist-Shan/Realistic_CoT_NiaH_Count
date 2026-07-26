#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 1 ]]; then
  echo "Usage: $0 RUN_ROOT" >&2
  exit 2
fi

run_root="$(readlink -f "$1")"
repo="${REALISTIC_NIAH_REPO_ROOT:-/lambda/nfs/Twist-CoT-Count-Multi-Model-v2/code/Realistic_CoT_NiaH_Count_v2_1}"
python_bin="${REALISTIC_NIAH_PYTHON:-/home/ubuntu/venvs/realistic-niah-vllm/bin/python}"
state_root="${run_root}/orchestration/shard_state"
status_file="${run_root}/orchestration/finalizer_status.tsv"

case "${run_root}" in
  /lambda/nfs/Twist-CoT-Count-Multi-Model-v2/runs/realistic_niah_v2_1/enumeration_rerun_and_gemma12_direct_appendix_*)
    ;;
  *)
    echo "Refusing unexpected run root: ${run_root}" >&2
    exit 2
    ;;
esac

mkdir -p "${state_root}/completed" "${state_root}/failed"

printf "checked_at_utc\tcompleted\tfailed\tstatus\n" > "${status_file}"
while true; do
  completed="$(
    find "${state_root}/completed" -maxdepth 1 -type f -name '*.tsv' \
      | wc -l
  )"
  failed="$(
    find "${state_root}/failed" -maxdepth 1 -type f -name '*.tsv' \
      | wc -l
  )"
  if [[ "${failed}" -gt 0 ]]; then
    printf "%s\t%s\t%s\tfailed\n" \
      "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "${completed}" "${failed}" \
      >> "${status_file}"
    exit 1
  fi
  if [[ "${completed}" -eq 15 ]]; then
    printf "%s\t%s\t%s\tmerging\n" \
      "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "${completed}" "${failed}" \
      >> "${status_file}"
    break
  fi
  printf "%s\t%s\t%s\twaiting\n" \
    "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "${completed}" "${failed}" \
    >> "${status_file}"
  sleep 30
done

(
  cd "${repo}"
  PYTHONPATH=src "${python_bin}" \
    scripts/merge_realistic_niah_prompt_revision_v2_1_shards.py \
    --run-root "${run_root}"
) > "${run_root}/orchestration/final_merge.log" 2>&1

printf "%s\t15\t0\tcompleted\n" \
  "$(date -u +%Y-%m-%dT%H:%M:%SZ)" >> "${status_file}"
