#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 || $# -gt 2 ]]; then
  echo "Usage: $0 RUN_ROOT [WORKERS]" >&2
  exit 2
fi

run_root="$(readlink -f "$1")"
requested_workers="${2:-}"
repo="${REALISTIC_NIAH_REPO_ROOT:-/lambda/nfs/Twist-CoT-Count-Multi-Model-v3/code/Realistic_CoT_NiaH_Count}"
python_bin="${REALISTIC_NIAH_PYTHON:-/home/ubuntu/venvs/realistic-niah-vllm/bin/python}"

test -x "${python_bin}"
test -d "${repo}/.git"
test -z "$(git -C "${repo}" status --short)"
case "${run_root}" in
  */runs/realistic_niah_v3/*)
    ;;
  *)
    echo "RUN_ROOT must be inside runs/realistic_niah_v3/" >&2
    exit 2
    ;;
esac

gpu_count="$(nvidia-smi --query-gpu=index --format=csv,noheader | wc -l)"
if [[ "${gpu_count}" -lt 1 ]]; then
  echo "At least one visible NVIDIA GPU is required" >&2
  exit 2
fi
workers="${requested_workers:-${gpu_count}}"
if ! [[ "${workers}" =~ ^[1-9][0-9]*$ ]] \
  || [[ "${workers}" -gt "${gpu_count}" ]] \
  || [[ "${workers}" -gt 8 ]]; then
  echo "WORKERS must be from 1 through min(visible GPUs, 8)" >&2
  exit 2
fi

(
  cd "${repo}"
  PYTHONPATH=src "${python_bin}" scripts/prepare_realistic_niah_v3.py \
    --run-root "${run_root}" \
    --repo-root "${repo}"
)
"${python_bin}" -c \
  'import json,sys; a=json.load(open(sys.argv[1],encoding="utf-8")); assert a["passed"] is True and not a["git"]["dirty"]' \
  "${run_root}/orchestration/prepare_audit.json"

run_tag="$(basename "${run_root}" | sed -E 's/[^A-Za-z0-9]+/-/g' | tail -c 25)"
worker_prefix="rniah-v3-${run_tag}-gpu"
finalizer_session="rniah-v3-${run_tag}-final"
for gpu_id in $(seq 0 "$((workers - 1))"); do
  session="${worker_prefix}${gpu_id}"
  if tmux has-session -t "${session}" 2>/dev/null; then
    echo "Worker tmux already exists: ${session}" >&2
    exit 2
  fi
done
if tmux has-session -t "${finalizer_session}" 2>/dev/null; then
  echo "Finalizer tmux already exists: ${finalizer_session}" >&2
  exit 2
fi

for gpu_id in $(seq 0 "$((workers - 1))"); do
  session="${worker_prefix}${gpu_id}"
  command="cd $(printf '%q' "${repo}") && exec bash scripts/run_realistic_niah_v3_worker.sh $(printf '%q' "${run_root}") ${gpu_id}"
  tmux new-session -d -s "${session}" "${command}"
done
finalizer_log="${run_root}/orchestration/finalizer.log"
finalizer_command="cd $(printf '%q' "${repo}") && exec bash scripts/finalize_realistic_niah_v3.sh $(printf '%q' "${run_root}") >> $(printf '%q' "${finalizer_log}") 2>&1"
tmux new-session -d -s "${finalizer_session}" "${finalizer_command}"

echo "Launched Realistic NIAH V3 with ${workers} worker(s)."
tmux list-sessions -F '#{session_name}' \
  | grep -E "^${worker_prefix}|^${finalizer_session}$" \
  | sort
