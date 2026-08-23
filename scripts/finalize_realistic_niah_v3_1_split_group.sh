#!/usr/bin/env bash
set -euo pipefail

[[ $# -eq 3 ]] || {
  echo "Usage: $0 RUN_ROOT SPLIT_RUN_ID ROLE" >&2
  exit 2
}
run_root="$(readlink -f "$1")"
split_run_id="$2"
role="$3"
repo="${REALISTIC_NIAH_REPO_ROOT:?Set REALISTIC_NIAH_REPO_ROOT}"
python_bin="${REALISTIC_NIAH_PYTHON:?Set REALISTIC_NIAH_PYTHON}"

case "${run_root}" in
  */runs/realistic_niah_v3_1/*) ;;
  *) echo "Refusing unexpected V3.1 run root: ${run_root}" >&2; exit 2 ;;
esac
[[ "${split_run_id}" =~ ^[A-Za-z0-9_.-]+$ ]] \
  || { echo "Invalid split run ID" >&2; exit 2; }
[[ "${role}" == "gemma4" || "${role}" == "general4" ]] \
  || { echo "ROLE must be gemma4 or general4" >&2; exit 2; }
command -v flock >/dev/null 2>&1 \
  || { echo "flock is required for split finalization" >&2; exit 2; }

coordination_dir="${run_root}/orchestration/split_runs/${split_run_id}"
mkdir -p "${coordination_dir}"
marker="${coordination_dir}/${role}.done"
temporary="$(mktemp "${coordination_dir}/.${role}.XXXXXX")"
{
  printf 'role=%s\n' "${role}"
  printf 'job_id=%s\n' "${SLURM_JOB_ID:-unknown}"
  printf 'completed_at_utc=%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
} > "${temporary}"
mv -f -- "${temporary}" "${marker}"

exec 9>"${coordination_dir}/finalize.lock"
flock -x 9
if [[ ! -s "${coordination_dir}/gemma4.done" \
   || ! -s "${coordination_dir}/general4.done" ]]; then
  printf 'Split group %s completed; peer group is still pending.\n' "${role}"
  exit 0
fi

final_audit="${run_root}/orchestration/final_shard_audit.json"
if [[ -s "${final_audit}" ]] \
  && "${python_bin}" -c \
    'import json,sys; a=json.load(open(sys.argv[1])); assert a["passed"] is True; assert a["requests"]==a["unique_request_ids"]==161280' \
    "${final_audit}" >/dev/null 2>&1; then
  printf 'Final shard audit already passed; no merge is repeated.\n'
  exit 0
fi

printf 'Both split groups completed; %s is running the single finalizer.\n' "${role}"
cd "${repo}"
bash scripts/finalize_realistic_niah_v3_1.sh "${run_root}"
