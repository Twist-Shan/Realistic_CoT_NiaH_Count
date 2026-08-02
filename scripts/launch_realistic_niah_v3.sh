#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 || $# -gt 2 ]]; then
  echo "Usage: $0 RUN_ROOT [MAX_GPUS]" >&2
  exit 2
fi

run_root="$(readlink -f "$1")"
requested_gpus="${2:-}"
repo="${REALISTIC_NIAH_REPO_ROOT:-/lambda/nfs/Twist-CoT-Count-Multi-Model-v3/code/Realistic_CoT_NiaH_Count}"
python_bin="${REALISTIC_NIAH_PYTHON:-/home/ubuntu/venvs/realistic-niah-vllm/bin/python}"

test -x "${python_bin}"
test -d "${repo}/.git"
test -z "$(git -C "${repo}" status --short)"
"${python_bin}" -c \
  'from importlib.metadata import version; from packaging.version import Version; assert version("transformers") == "5.14.1"; assert version("vllm") == "0.25.1"; assert Version(version("mistral-common")) >= Version("1.8.6")'
case "${run_root}" in
  */runs/realistic_niah_v3/*) ;;
  *)
    echo "RUN_ROOT must be inside runs/realistic_niah_v3/" >&2
    exit 2
    ;;
esac

gpu_count="$(nvidia-smi --query-gpu=index --format=csv,noheader | wc -l)"
if [[ "${gpu_count}" -lt 2 ]]; then
  echo "V3 formal scheduling requires at least two visible GPUs for Gemma4-31B" >&2
  exit 2
fi
max_gpus="${requested_gpus:-${gpu_count}}"
if ! [[ "${max_gpus}" =~ ^[1-9][0-9]*$ ]] \
  || [[ "${max_gpus}" -lt 2 ]] \
  || [[ "${max_gpus}" -gt "${gpu_count}" ]] \
  || [[ "${max_gpus}" -gt 8 ]]; then
  echo "MAX_GPUS must be from 2 through min(visible GPUs, 8)" >&2
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

mkdir -p \
  "${run_root}/orchestration/shard_state/completed" \
  "${run_root}/orchestration/shard_state/failed"
run_tag="$(basename "${run_root}" | sed -E 's/[^A-Za-z0-9]+/-/g' | tail -c 25)"
scheduler_session="rniah-v3-${run_tag}-scheduler"
finalizer_session="rniah-v3-${run_tag}-final"
for session in "${scheduler_session}" "${finalizer_session}"; do
  if tmux has-session -t "${session}" 2>/dev/null; then
    echo "V3 tmux already exists: ${session}" >&2
    exit 2
  fi
done

scheduler_log="${run_root}/orchestration/scheduler.log"
scheduler_command="cd $(printf '%q' "${repo}") && PYTHONPATH=src exec $(printf '%q' "${python_bin}") scripts/schedule_realistic_niah_v3.py --run-root $(printf '%q' "${run_root}") --repo-root $(printf '%q' "${repo}") --max-gpus ${max_gpus} >> $(printf '%q' "${scheduler_log}") 2>&1"
tmux new-session -d -s "${scheduler_session}" "${scheduler_command}"

finalizer_log="${run_root}/orchestration/finalizer.log"
finalizer_command="cd $(printf '%q' "${repo}") && exec bash scripts/finalize_realistic_niah_v3.sh $(printf '%q' "${run_root}") >> $(printf '%q' "${finalizer_log}") 2>&1"
tmux new-session -d -s "${finalizer_session}" "${finalizer_command}"

echo "Launched Realistic NIAH V3 resource-aware scheduling on ${max_gpus} GPU(s)."
echo "Gemma4-31B shards reserve two GPUs with tensor parallelism; other shards use one."
tmux list-sessions -F '#{session_name}' \
  | grep -E "^rniah-v3-${run_tag}-(scheduler|final)$"
