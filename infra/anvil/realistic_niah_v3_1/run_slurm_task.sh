#!/usr/bin/env bash
set -euo pipefail

[[ $# -eq 1 ]] || { echo "Usage: $0 RUN_ROOT" >&2; exit 2; }
[[ -n "${SLURM_JOB_ID:-}" && -n "${SLURM_PROCID:-}" ]] \
  || { echo "run_slurm_task.sh requires an srun task context" >&2; exit 2; }
[[ -n "${CUDA_VISIBLE_DEVICES:-}" ]] \
  || { echo "Slurm did not bind a GPU to task ${SLURM_PROCID}" >&2; exit 2; }
[[ -n "${CUDA_HOME:-}" && -n "${LD_LIBRARY_PATH:-}" ]] \
  || { echo "CUDA runtime environment did not reach task ${SLURM_PROCID}" >&2; exit 2; }
[[ -e "${CUDA_HOME}/lib64/libcudart.so.12" ]] \
  || { echo "CUDA runtime is unavailable on task ${SLURM_PROCID}" >&2; exit 2; }
: "${REALISTIC_NIAH_PYTHON:?Set REALISTIC_NIAH_PYTHON}"
"${REALISTIC_NIAH_PYTHON}" -c \
  'import ctypes; ctypes.CDLL("libcudart.so.12")' \
  || { echo "CUDA runtime cannot be loaded on task ${SLURM_PROCID}" >&2; exit 2; }

run_root="$(readlink -f "$1")"
repo="${REALISTIC_NIAH_REPO_ROOT:?Set REALISTIC_NIAH_REPO_ROOT}"
export REALISTIC_NIAH_DEVICE_MODE=allocated
export REALISTIC_NIAH_WORKER_ID="slurm-${SLURM_JOB_ID}-task-${SLURM_PROCID}"
export REALISTIC_NIAH_STAGGER_SLOT="${SLURM_PROCID}"

printf 'worker=%s host=%s local_task=%s cuda_visible_devices=%s cuda_home=%s\n' \
  "${REALISTIC_NIAH_WORKER_ID}" "$(hostname)" "${SLURM_LOCALID:-}" \
  "${CUDA_VISIBLE_DEVICES}" "${CUDA_HOME}"
exec bash "${repo}/scripts/run_realistic_niah_v3_1_worker.sh" \
  "${run_root}" "${SLURM_PROCID}"
