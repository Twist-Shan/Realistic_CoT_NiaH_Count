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
case "${run_root}" in
  */runs/realistic_niah_v3_1/*) ;;
  *) echo "RUN_ROOT must be inside runs/realistic_niah_v3_1/" >&2; exit 2 ;;
esac
test -x "${python_bin}"
test -d "${repo}/.git"
test -z "$(git -C "${repo}" status --short)"
gpu_count="$(nvidia-smi --query-gpu=index --format=csv,noheader | wc -l)"
workers="${requested_workers:-${gpu_count}}"
[[ "${workers}" =~ ^[1-9][0-9]*$ ]] || { echo "Invalid WORKERS" >&2; exit 2; }
[[ "${workers}" -le "${gpu_count}" && "${workers}" -le 8 ]] \
  || { echo "WORKERS exceeds visible GPUs or 8" >&2; exit 2; }

(
  cd "${repo}"
  PYTHONPATH=src "${python_bin}" scripts/prepare_realistic_niah_v3_1.py \
    --run-root "${run_root}" --repo-root "${repo}"
)
run_tag="$(basename "${run_root}" | sed -E 's/[^A-Za-z0-9]+/-/g' | tail -c 25)"
worker_prefix="rniah-v31-${run_tag}-gpu"
finalizer_session="rniah-v31-${run_tag}-final"
for gpu_id in $(seq 0 "$((workers - 1))"); do
  session="${worker_prefix}${gpu_id}"
  tmux has-session -t "${session}" 2>/dev/null \
    && { echo "Existing tmux session: ${session}" >&2; exit 2; }
done
tmux has-session -t "${finalizer_session}" 2>/dev/null \
  && { echo "Existing tmux session: ${finalizer_session}" >&2; exit 2; }
for gpu_id in $(seq 0 "$((workers - 1))"); do
  session="${worker_prefix}${gpu_id}"
  command="cd $(printf '%q' "${repo}") && exec bash scripts/run_realistic_niah_v3_1_worker.sh $(printf '%q' "${run_root}") ${gpu_id}"
  tmux new-session -d -s "${session}" "${command}"
done
finalizer_log="${run_root}/orchestration/finalizer.log"
finalizer_command="cd $(printf '%q' "${repo}") && exec bash scripts/finalize_realistic_niah_v3_1.sh $(printf '%q' "${run_root}") >> $(printf '%q' "${finalizer_log}") 2>&1"
tmux new-session -d -s "${finalizer_session}" "${finalizer_command}"
echo "Launched V3.1 with ${workers} worker(s)."
