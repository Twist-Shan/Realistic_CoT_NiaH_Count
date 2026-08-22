#!/usr/bin/env bash
set -eo pipefail

# Anvil's module bootstrap reads optional shell variables such as ROOTSAFE.
# Load it before enabling nounset so a clean Slurm environment cannot fail
# before the experiment's own validation starts.
source /etc/profile.d/modules.sh
set -u

[[ $# -eq 1 ]] || { echo "Usage: $0 RUN_ROOT" >&2; exit 2; }
[[ -n "${SLURM_JOB_ID:-}" && -n "${SLURM_NTASKS:-}" ]] \
  || { echo "This script must run inside a Slurm allocation" >&2; exit 2; }

run_root="$(readlink -f "$1")"
repo="${REALISTIC_NIAH_REPO_ROOT:?Set REALISTIC_NIAH_REPO_ROOT}"
python_bin="${REALISTIC_NIAH_PYTHON:?Set REALISTIC_NIAH_PYTHON}"
hf_cache="${REALISTIC_NIAH_HF_CACHE:?Set REALISTIC_NIAH_HF_CACHE}"
expected_commit="${REALISTIC_NIAH_EXPECTED_COMMIT:?Set REALISTIC_NIAH_EXPECTED_COMMIT}"
tasks_per_node="${REALISTIC_NIAH_TASKS_PER_NODE:-4}"
task_script="${repo}/infra/anvil/realistic_niah_v3_1/run_slurm_task.sh"

case "${run_root}" in
  */runs/realistic_niah_v3_1/*) ;;
  *) echo "Refusing unexpected V3.1 run root: ${run_root}" >&2; exit 2 ;;
esac
[[ -x "${python_bin}" ]] || { echo "Python is not executable: ${python_bin}" >&2; exit 2; }
[[ -d "${hf_cache}" ]] || { echo "HF cache does not exist: ${hf_cache}" >&2; exit 2; }
[[ -r "${task_script}" ]] \
  || { echo "Shared worker launcher is not readable: ${task_script}" >&2; exit 2; }
[[ -z "$(git -C "${repo}" status --short)" ]] \
  || { echo "Formal V3.1 execution requires a clean Git worktree" >&2; exit 2; }
actual_commit="$(git -C "${repo}" rev-parse HEAD)"
[[ "${actual_commit}" == "${expected_commit}" ]] \
  || { echo "Formal V3.1 commit mismatch: ${actual_commit} != ${expected_commit}" >&2; exit 2; }

module --force purge
module load modtree/gpu
module load conda

# The restricted Slurm export intentionally does not activate the conda
# environment.  vLLM launches tools such as ninja by executable name, so make
# the frozen environment's executables discoverable before runtime preflight.
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
  || { echo "ninja is unavailable from the frozen environment: ${env_bin}" >&2; exit 2; }
printf '__global__ void realistic_niah_h100_preflight(){}\n' \
  | "${CUDA_HOME}/bin/nvcc" -x cu \
      -gencode=arch=compute_90a,code=sm_90a -c -o /dev/null - \
  || { echo "CUDA compiler does not support H100 compute_90a: ${CUDA_HOME}" >&2; exit 2; }

# vLLM 0.25.1 enables FlashInfer sampling by default.  On Anvil this can
# trigger a JIT build for compute_90a with the site nvcc, even though the
# frozen PyTorch wheel already carries its own CUDA runtime.  Use vLLM's
# documented native-sampler fallback so H100 execution does not depend on a
# compatible system CUDA compiler.
export VLLM_USE_FLASHINFER_SAMPLER=0

"${python_bin}" -c \
  'import sys; from importlib.metadata import version; from packaging.version import Version; assert sys.version_info[:2] == (3, 11); assert version("transformers") == "5.14.1"; assert version("vllm") == "0.25.1"; assert Version("1.8.6") <= Version(version("mistral-common")) < Version("2")'

slurm_dir="${run_root}/orchestration/slurm"
mkdir -p "${slurm_dir}"
provenance_file="${slurm_dir}/job-${SLURM_JOB_ID}-provenance.txt"
{
  printf 'started_at_utc=%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  printf 'job_id=%s\njob_name=%s\nnodes=%s\ntasks=%s\ntasks_per_node=%s\n' \
    "${SLURM_JOB_ID}" "${SLURM_JOB_NAME:-}" "${SLURM_NNODES:-}" \
    "${SLURM_NTASKS}" "${tasks_per_node}"
  printf 'node_list=%s\npartition=%s\naccount=%s\n' \
    "${SLURM_JOB_NODELIST:-}" "${SLURM_JOB_PARTITION:-}" "${SLURM_JOB_ACCOUNT:-}"
  printf 'repo=%s\ncommit=%s\npython=%s\nhf_cache=%s\n' \
    "${repo}" "$(git -C "${repo}" rev-parse HEAD)" "${python_bin}" "${hf_cache}"
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

srun \
  --ntasks="${SLURM_NTASKS}" \
  --ntasks-per-node="${tasks_per_node}" \
  --gpus-per-task=1 \
  --gpu-bind=single:1 \
  --cpu-bind=cores \
  --kill-on-bad-exit=1 \
  --export=HOME,USER,PATH,SHELL,REALISTIC_NIAH_REPO_ROOT,REALISTIC_NIAH_PYTHON,REALISTIC_NIAH_HF_CACHE,REALISTIC_NIAH_TASKS_PER_NODE,REALISTIC_NIAH_EXPECTED_COMMIT,CUDA_HOME,CUDA_PATH,LD_LIBRARY_PATH,VLLM_USE_FLASHINFER_SAMPLER \
  --label \
  bash "${task_script}" "${run_root}"

bash scripts/finalize_realistic_niah_v3_1.sh "${run_root}"
printf 'completed_at_utc=%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" \
  >> "${provenance_file}"
printf 'V3.1 inference and merge completed for %s\n' "${run_root}"
