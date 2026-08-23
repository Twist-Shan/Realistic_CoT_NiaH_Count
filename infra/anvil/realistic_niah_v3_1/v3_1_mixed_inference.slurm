#!/usr/bin/env bash
set -eo pipefail

source /etc/profile.d/modules.sh
set -u

[[ $# -eq 1 ]] || { echo "Usage: $0 RUN_ROOT" >&2; exit 2; }
[[ -n "${SLURM_JOB_ID:-}" && -n "${SLURM_NNODES:-}" ]] \
  || { echo "This script must run inside a Slurm allocation" >&2; exit 2; }
[[ "${SLURM_NNODES}" == "2" ]] \
  || { echo "Mixed V3.1 execution requires exactly two H100 nodes" >&2; exit 2; }

run_root="$(readlink -f "$1")"
repo="${REALISTIC_NIAH_REPO_ROOT:?Set REALISTIC_NIAH_REPO_ROOT}"
python_bin="${REALISTIC_NIAH_PYTHON:?Set REALISTIC_NIAH_PYTHON}"
hf_cache="${REALISTIC_NIAH_HF_CACHE:?Set REALISTIC_NIAH_HF_CACHE}"
expected_commit="${REALISTIC_NIAH_EXPECTED_COMMIT:?Set REALISTIC_NIAH_EXPECTED_COMMIT}"
task_script="${repo}/infra/anvil/realistic_niah_v3_1/run_slurm_task.sh"

case "${run_root}" in
  */runs/realistic_niah_v3_1/*) ;;
  *) echo "Refusing unexpected V3.1 run root: ${run_root}" >&2; exit 2 ;;
esac
[[ -x "${python_bin}" ]] || { echo "Python is not executable: ${python_bin}" >&2; exit 2; }
[[ -d "${hf_cache}" ]] || { echo "HF cache does not exist: ${hf_cache}" >&2; exit 2; }
[[ -r "${task_script}" ]] || { echo "Missing worker launcher: ${task_script}" >&2; exit 2; }
[[ -z "$(git -C "${repo}" status --short)" ]] \
  || { echo "Formal V3.1 execution requires a clean Git worktree" >&2; exit 2; }
actual_commit="$(git -C "${repo}" rev-parse HEAD)"
[[ "${actual_commit}" == "${expected_commit}" ]] \
  || { echo "Formal V3.1 commit mismatch: ${actual_commit} != ${expected_commit}" >&2; exit 2; }

mapfile -t allocation_nodes < <(scontrol show hostnames "${SLURM_JOB_NODELIST}")
[[ "${#allocation_nodes[@]}" -eq 2 ]] \
  || { echo "Unable to resolve the two allocation nodes" >&2; exit 2; }
node_a="${allocation_nodes[0]}"
node_b="${allocation_nodes[1]}"

module --force purge
module load modtree/gpu
module load conda

env_bin="$(dirname -- "${python_bin}")"
cuda_home="${REALISTIC_NIAH_CUDA_HOME:-/apps/anvilgpu/external/apps/cuda-toolkit/12.8.0}"
[[ -x "${cuda_home}/bin/nvcc" ]] \
  || { echo "CUDA compiler is unavailable: ${cuda_home}/bin/nvcc" >&2; exit 2; }
export CUDA_HOME="${cuda_home}"
export CUDA_PATH="${cuda_home}"
export PATH="${env_bin}:${CUDA_HOME}/bin:${PATH}"
[[ -e "${CUDA_HOME}/lib64/libcudart.so.12" ]] \
  || { echo "CUDA runtime is unavailable: ${CUDA_HOME}/lib64/libcudart.so.12" >&2; exit 2; }
export LD_LIBRARY_PATH="${CUDA_HOME}/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
command -v ninja >/dev/null 2>&1 \
  || { echo "ninja is unavailable from the frozen environment" >&2; exit 2; }
printf '__global__ void realistic_niah_h100_preflight(){}\n' \
  | "${CUDA_HOME}/bin/nvcc" -x cu \
      -gencode=arch=compute_90a,code=sm_90a -c -o /dev/null -
export VLLM_USE_FLASHINFER_SAMPLER=0

"${python_bin}" -c \
  'import sys; from importlib.metadata import version; from packaging.version import Version; assert sys.version_info[:2] == (3, 11); assert version("transformers") == "5.14.1"; assert version("vllm") == "0.25.1"; assert Version("1.8.6") <= Version(version("mistral-common")) < Version("2")'

slurm_dir="${run_root}/orchestration/slurm"
mkdir -p "${slurm_dir}"
provenance_file="${slurm_dir}/job-${SLURM_JOB_ID}-provenance.txt"
{
  printf 'started_at_utc=%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  printf 'job_id=%s\njob_name=%s\nlayout=mixed_8gpu\nnode_a=%s\nnode_b=%s\n' \
    "${SLURM_JOB_ID}" "${SLURM_JOB_NAME:-}" "${node_a}" "${node_b}"
  printf 'gemma_gpus=2\ngeneral_initial_gpus=6\ngeneral_topup_gpus=2\n'
  printf 'repo=%s\ncommit=%s\npython=%s\nhf_cache=%s\n' \
    "${repo}" "${actual_commit}" "${python_bin}" "${hf_cache}"
  scontrol show job "${SLURM_JOB_ID}"
  module list
  printf 'cuda_home=%s\nnvcc=%s\nld_library_path=%s\n' \
    "${CUDA_HOME}" "$(command -v nvcc)" "${LD_LIBRARY_PATH}"
  nvcc --version
  "${python_bin}" -c \
    'import platform, torch, transformers, vllm; print({"python": platform.python_version(), "torch": torch.__version__, "transformers": transformers.__version__, "vllm": vllm.__version__, "cuda": torch.version.cuda})'
  nvidia-smi --query-gpu=name,uuid,memory.total,driver_version --format=csv
} > "${provenance_file}" 2>&1

cd "${repo}"
PYTHONPATH=src "${python_bin}" scripts/prepare_realistic_niah_v3_1.py \
  --run-root "${run_root}" --repo-root "${repo}" \
  --expected-commit "${expected_commit}" --require-source-revision

runtime_exports="HOME,USER,PATH,SHELL,REALISTIC_NIAH_REPO_ROOT,REALISTIC_NIAH_PYTHON,REALISTIC_NIAH_HF_CACHE,REALISTIC_NIAH_EXPECTED_COMMIT,CUDA_HOME,CUDA_PATH,LD_LIBRARY_PATH,VLLM_USE_FLASHINFER_SAMPLER"
children=()
cleanup_children() {
  local status=$?
  trap - EXIT
  if [[ "${status}" -ne 0 ]]; then
    for child in "${children[@]}"; do
      kill "${child}" 2>/dev/null || true
    done
    for child in "${children[@]}"; do
      wait "${child}" 2>/dev/null || true
    done
  fi
  exit "${status}"
}
trap cleanup_children EXIT

# Node A: Gemma uses two GPUs with TP=2; two single-GPU workers use the rest.
srun --exact --exclusive --nodes=1 --nodelist="${node_a}" \
  --ntasks=2 --ntasks-per-node=2 --cpus-per-task=12 \
  --gpus-per-task=1 --gpu-bind=per_task:1 --mem=240G \
  --kill-on-bad-exit=1 \
  --export="${runtime_exports},REALISTIC_NIAH_TASKS_PER_NODE=2,REALISTIC_NIAH_GPUS_PER_TASK=1,REALISTIC_NIAH_MODEL_EXCLUDE=Gemma4-31B,REALISTIC_NIAH_FINALIZE=0,REALISTIC_NIAH_TENSOR_PARALLEL_SIZE=1,REALISTIC_NIAH_WORKER_NAMESPACE=general-a,REALISTIC_NIAH_WORKER_OFFSET=0" \
  --label bash "${task_script}" "${run_root}" &
general_a_pid=$!
children+=("${general_a_pid}")

# Node B: four independent single-GPU workers.
srun --exact --exclusive --nodes=1 --nodelist="${node_b}" \
  --ntasks=4 --ntasks-per-node=4 --cpus-per-task=12 \
  --gpus-per-task=1 --gpu-bind=per_task:1 --mem=480G \
  --kill-on-bad-exit=1 \
  --export="${runtime_exports},REALISTIC_NIAH_TASKS_PER_NODE=4,REALISTIC_NIAH_GPUS_PER_TASK=1,REALISTIC_NIAH_MODEL_EXCLUDE=Gemma4-31B,REALISTIC_NIAH_FINALIZE=0,REALISTIC_NIAH_TENSOR_PARALLEL_SIZE=1,REALISTIC_NIAH_WORKER_NAMESPACE=general-b,REALISTIC_NIAH_WORKER_OFFSET=2" \
  --label bash "${task_script}" "${run_root}" &
general_b_pid=$!
children+=("${general_b_pid}")

srun --exact --exclusive --nodes=1 --nodelist="${node_a}" \
  --ntasks=1 --ntasks-per-node=1 --cpus-per-task=24 \
  --gpus-per-task=2 --gpu-bind=per_task:2 --mem=240G \
  --kill-on-bad-exit=1 \
  --export="${runtime_exports},REALISTIC_NIAH_TASKS_PER_NODE=1,REALISTIC_NIAH_GPUS_PER_TASK=2,REALISTIC_NIAH_MODEL_FILTER=Gemma4-31B,REALISTIC_NIAH_FINALIZE=0,REALISTIC_NIAH_TENSOR_PARALLEL_SIZE=2,REALISTIC_NIAH_WORKER_NAMESPACE=gemma,REALISTIC_NIAH_WORKER_OFFSET=0" \
  --label bash "${task_script}" "${run_root}" &
gemma_pid=$!
children+=("${gemma_pid}")

gemma_status=0
wait "${gemma_pid}" || gemma_status=$?
children=("${general_a_pid}" "${general_b_pid}")
[[ "${gemma_status}" -eq 0 ]] \
  || { echo "Gemma4-31B TP=2 step failed with ${gemma_status}" >&2; exit "${gemma_status}"; }

# Reuse Gemma's two GPUs as ordinary workers after its bundle is complete.
srun --exact --exclusive --nodes=1 --nodelist="${node_a}" \
  --ntasks=2 --ntasks-per-node=2 --cpus-per-task=12 \
  --gpus-per-task=1 --gpu-bind=per_task:1 --mem=240G \
  --kill-on-bad-exit=1 \
  --export="${runtime_exports},REALISTIC_NIAH_TASKS_PER_NODE=2,REALISTIC_NIAH_GPUS_PER_TASK=1,REALISTIC_NIAH_MODEL_EXCLUDE=Gemma4-31B,REALISTIC_NIAH_FINALIZE=0,REALISTIC_NIAH_TENSOR_PARALLEL_SIZE=1,REALISTIC_NIAH_WORKER_NAMESPACE=topup,REALISTIC_NIAH_WORKER_OFFSET=6" \
  --label bash "${task_script}" "${run_root}" &
topup_pid=$!
children+=("${topup_pid}")

general_status=0
wait "${general_a_pid}" || general_status=$?
wait "${general_b_pid}" || general_status=$?
wait "${topup_pid}" || general_status=$?
children=()
[[ "${general_status}" -eq 0 ]] \
  || { echo "One or more single-GPU worker steps failed" >&2; exit "${general_status}"; }

bash scripts/finalize_realistic_niah_v3_1.sh "${run_root}"
printf 'completed_at_utc=%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" \
  >> "${provenance_file}"
printf 'Mixed 8-GPU V3.1 inference and merge completed for %s\n' "${run_root}"
