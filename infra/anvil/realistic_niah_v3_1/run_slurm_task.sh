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
gpus_per_task="${REALISTIC_NIAH_GPUS_PER_TASK:-1}"
tensor_parallel_size="${REALISTIC_NIAH_TENSOR_PARALLEL_SIZE:-1}"
[[ "${gpus_per_task}" =~ ^[1-9][0-9]*$ ]] \
  || { echo "REALISTIC_NIAH_GPUS_PER_TASK must be positive" >&2; exit 2; }
[[ "${tensor_parallel_size}" =~ ^[1-9][0-9]*$ ]] \
  || { echo "REALISTIC_NIAH_TENSOR_PARALLEL_SIZE must be positive" >&2; exit 2; }
IFS=',' read -r -a visible_gpus <<< "${CUDA_VISIBLE_DEVICES}"
[[ "${#visible_gpus[@]}" -eq "${gpus_per_task}" ]] \
  || { echo "Visible GPU count does not match GPUs per task" >&2; exit 2; }
[[ "${tensor_parallel_size}" -eq "${gpus_per_task}" ]] \
  || { echo "Tensor parallel size does not match GPUs per task" >&2; exit 2; }
if [[ "${tensor_parallel_size}" -gt 1 ]]; then
  "${REALISTIC_NIAH_PYTHON}" -c \
    'import dataclasses; from importlib.metadata import version; from vllm.engine.arg_utils import EngineArgs; names={field.name for field in dataclasses.fields(EngineArgs)}; required={"tensor_parallel_size","gpu_memory_utilization","enforce_eager","disable_custom_all_reduce"}; assert version("vllm")=="0.25.1"; assert required <= names; args=EngineArgs(model="Qwen/Qwen3-4B", tensor_parallel_size=2, gpu_memory_utilization=0.92, enforce_eager=True, disable_custom_all_reduce=True); assert (args.tensor_parallel_size,args.gpu_memory_utilization,args.enforce_eager,args.disable_custom_all_reduce)==(2,0.92,True,True); print("VLLM_TP2_PREFLIGHT_OK")' \
    || { echo "vLLM TP=2 stability preflight failed" >&2; exit 2; }
fi

run_root="$(readlink -f "$1")"
repo="${REALISTIC_NIAH_REPO_ROOT:?Set REALISTIC_NIAH_REPO_ROOT}"
worker_namespace="${REALISTIC_NIAH_WORKER_NAMESPACE:-task}"
worker_offset="${REALISTIC_NIAH_WORKER_OFFSET:-0}"
[[ "${worker_namespace}" =~ ^[A-Za-z0-9_.-]+$ ]] \
  || { echo "REALISTIC_NIAH_WORKER_NAMESPACE is invalid" >&2; exit 2; }
[[ "${worker_offset}" =~ ^[0-9]+$ ]] \
  || { echo "REALISTIC_NIAH_WORKER_OFFSET must be non-negative" >&2; exit 2; }
worker_slot="$((worker_offset + SLURM_PROCID))"
export REALISTIC_NIAH_DEVICE_MODE=allocated
export REALISTIC_NIAH_WORKER_ID="slurm-${SLURM_JOB_ID}-${worker_namespace}-${worker_slot}"
export REALISTIC_NIAH_STAGGER_SLOT="${worker_slot}"

printf 'worker=%s host=%s local_task=%s cuda_visible_devices=%s cuda_home=%s\n' \
  "${REALISTIC_NIAH_WORKER_ID}" "$(hostname)" "${SLURM_LOCALID:-}" \
  "${CUDA_VISIBLE_DEVICES}" "${CUDA_HOME}"
exec bash "${repo}/scripts/run_realistic_niah_v3_1_worker.sh" \
  "${run_root}" "${worker_slot}"
