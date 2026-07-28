#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 2 || $# -gt 3 ]]; then
  echo "Usage: $0 SOURCE_FORMAL_RUN_ROOT RUN_ROOT [WORKERS]" >&2
  exit 2
fi

source_formal_root="$(readlink -f "$1")"
run_root="$(readlink -m "$2")"
requested_workers="${3:-}"
repo="${REALISTIC_NIAH_REPO_ROOT:-/lambda/nfs/Twist-CoT-Count-Multi-Model-v2/code/Realistic_CoT_NiaH_Count}"
python_bin="${REALISTIC_NIAH_PYTHON:-/home/ubuntu/venvs/realistic-niah-vllm/bin/python}"

test -x "${python_bin}"
test -d "${repo}/.git"
test -z "$(git -C "${repo}" status --short)"
"${python_bin}" -c \
  'from importlib.metadata import version; import vllm; assert version("transformers") == "5.5.3"; assert version("vllm") == "0.25.1"'

gpu_count="$(nvidia-smi --query-gpu=index --format=csv,noheader | wc -l)"
if [[ "${gpu_count}" -lt 1 ]]; then
  echo "At least one visible NVIDIA GPU is required" >&2
  exit 2
fi
if [[ -z "${requested_workers}" ]]; then
  workers="${gpu_count}"
  if [[ "${workers}" -gt 2 ]]; then
    workers=2
  fi
else
  workers="${requested_workers}"
fi
if ! [[ "${workers}" =~ ^[1-9][0-9]*$ ]] \
  || [[ "${workers}" -gt "${gpu_count}" ]] \
  || [[ "${workers}" -gt 4 ]]; then
  echo "WORKERS must be from 1 through min(visible GPUs, 4)" >&2
  exit 2
fi

case "${run_root}" in
  */runs/realistic_niah_v2/olmo3_7b_extension_*)
    ;;
  *)
    echo "RUN_ROOT must end in realistic_niah_v2/olmo3_7b_extension_*" >&2
    exit 2
    ;;
esac

(
  cd "${repo}"
  PYTHONPATH=src "${python_bin}" \
    scripts/prepare_realistic_niah_olmo3_extension.py \
    --source-formal-run-root "${source_formal_root}" \
    --run-root "${run_root}" \
    --repo-root "${repo}"
)
"${python_bin}" -c \
  'import json,sys; a=json.load(open(sys.argv[1],encoding="utf-8")); assert a["passed"] is True and not a["git"]["dirty"]' \
  "${run_root}/orchestration/prepare_audit.json"

run_tag="$(basename "${run_root}" | sed -E 's/[^A-Za-z0-9]+/-/g' | tail -c 25)"
worker_prefix="rniah-olmo3-${run_tag}-gpu"
finalizer_session="rniah-olmo3-${run_tag}-final"
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
  command="cd $(printf '%q' "${repo}") && exec bash scripts/run_realistic_niah_olmo3_extension_worker.sh $(printf '%q' "${run_root}") ${gpu_id}"
  tmux new-session -d -s "${session}" "${command}"
done
finalizer_log="${run_root}/orchestration/finalizer.log"
finalizer_command="cd $(printf '%q' "${repo}") && exec bash scripts/finalize_realistic_niah_olmo3_extension.sh $(printf '%q' "${run_root}") >> $(printf '%q' "${finalizer_log}") 2>&1"
tmux new-session -d -s "${finalizer_session}" "${finalizer_command}"

echo "Launched OLMo 3 extension with ${workers} worker(s)."
tmux list-sessions -F '#{session_name}' \
  | grep -E "^${worker_prefix}|^${finalizer_session}$" \
  | sort
