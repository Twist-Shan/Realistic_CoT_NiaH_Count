#!/usr/bin/env bash
set -eo pipefail

source /etc/profile.d/modules.sh
set -u

[[ $# -eq 1 ]] || { echo "Usage: $0 RUN_ROOT" >&2; exit 2; }
[[ -n "${SLURM_JOB_ID:-}" && "${SLURM_NNODES:-}" == "1" ]] \
  || { echo "This job requires exactly one Slurm node" >&2; exit 2; }

run_root="$(readlink -f "$1")"
repo="${REALISTIC_NIAH_REPO_ROOT:?Set REALISTIC_NIAH_REPO_ROOT}"
python_bin="${REALISTIC_NIAH_PYTHON:?Set REALISTIC_NIAH_PYTHON}"
hf_cache="${REALISTIC_NIAH_HF_CACHE:?Set REALISTIC_NIAH_HF_CACHE}"
expected_commit="${REALISTIC_NIAH_EXPECTED_COMMIT:?Set REALISTIC_NIAH_EXPECTED_COMMIT}"
dataset_seal_sha256="${REALISTIC_NIAH_DATASET_SEAL_SHA256:?Set REALISTIC_NIAH_DATASET_SEAL_SHA256}"
resume_from_commits="${REALISTIC_NIAH_RESUME_FROM_COMMITS:-}"
model_label="${REALISTIC_NIAH_MODEL_LABEL:?Set REALISTIC_NIAH_MODEL_LABEL}"
worker_script="${repo}/infra/anvil/realistic_niah_v3_3_long_context/run_slurm_worker.sh"

case "${run_root}" in
  */runs/realistic_niah_v3_3_long_context/*) ;;
  *) echo "Refusing unexpected V3.3 long-context run root: ${run_root}" >&2; exit 2 ;;
esac
[[ "${expected_commit}" =~ ^[0-9a-f]{40}$ ]] \
  || { echo "Invalid expected commit" >&2; exit 2; }
[[ "${dataset_seal_sha256}" =~ ^[0-9a-f]{64}$ ]] \
  || { echo "Invalid dataset seal SHA256" >&2; exit 2; }
[[ -z "${resume_from_commits}" \
    || "${resume_from_commits}" =~ ^[0-9a-f]{40}(:[0-9a-f]{40})*$ ]] \
  || { echo "Invalid REALISTIC_NIAH_RESUME_FROM_COMMITS" >&2; exit 2; }
[[ "${model_label}" == "Gemma4-31B" || "${model_label}" == "Qwen3-32B" ]] \
  || { echo "Invalid long-context model: ${model_label}" >&2; exit 2; }
[[ -x "${python_bin}" ]] || { echo "Python is not executable" >&2; exit 2; }
[[ -d "${hf_cache}" ]] || { echo "HF cache does not exist" >&2; exit 2; }
[[ -r "${worker_script}" ]] || { echo "Worker script is missing" >&2; exit 2; }
command -v flock >/dev/null 2>&1 \
  || { echo "flock is required for finalization" >&2; exit 2; }
[[ -z "$(git -C "${repo}" status --short)" ]] \
  || { echo "Formal execution requires a clean Git worktree" >&2; exit 2; }
actual_commit="$(git -C "${repo}" rev-parse HEAD)"
[[ "${actual_commit}" == "${expected_commit}" ]] \
  || { echo "Commit mismatch: ${actual_commit} != ${expected_commit}" >&2; exit 2; }
export REALISTIC_NIAH_RESUME_FROM_COMMITS="${resume_from_commits}"

module --force purge
module load modtree/gpu
module load conda

env_bin="$(dirname -- "${python_bin}")"
cuda_home="${REALISTIC_NIAH_CUDA_HOME:-/apps/anvilgpu/external/apps/cuda-toolkit/12.8.0}"
[[ -x "${cuda_home}/bin/nvcc" ]] \
  || { echo "CUDA compiler is unavailable: ${cuda_home}/bin/nvcc" >&2; exit 2; }
[[ -e "${cuda_home}/lib64/libcudart.so.12" ]] \
  || { echo "CUDA runtime is unavailable" >&2; exit 2; }
export CUDA_HOME="${cuda_home}"
export CUDA_PATH="${cuda_home}"
export PATH="${env_bin}:${CUDA_HOME}/bin:${PATH}"
export LD_LIBRARY_PATH="${CUDA_HOME}/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
export VLLM_USE_FLASHINFER_SAMPLER=0
command -v ninja >/dev/null 2>&1 \
  || { echo "ninja is unavailable from the frozen environment" >&2; exit 2; }
printf '__global__ void realistic_niah_h100_preflight(){}\n' \
  | "${CUDA_HOME}/bin/nvcc" -x cu \
      -gencode=arch=compute_90a,code=sm_90a -c -o /dev/null -

"${python_bin}" -c \
  'import sys; from importlib.metadata import version; assert sys.version_info[:2] == (3, 11); assert version("transformers") == "5.14.1"; assert version("vllm") == "0.25.1"; assert version("torch") == "2.11.0"'

gpu_inventory="$(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader)"
[[ "$(printf '%s\n' "${gpu_inventory}" | sed '/^[[:space:]]*$/d' | wc -l)" -eq 4 ]] \
  || { echo "The allocation must expose exactly four GPUs" >&2; exit 2; }
if ! printf '%s\n' "${gpu_inventory}" \
  | awk -F, 'index($1, "H100") == 0 || ($2 + 0) < 80000 { exit 1 }'; then
  echo "Formal execution requires four H100 80GB GPUs: ${gpu_inventory}" >&2
  exit 2
fi

slurm_dir="${run_root}/orchestration/slurm"
worker_log_dir="${run_root}/orchestration/worker_logs"
mkdir -p "${slurm_dir}" "${worker_log_dir}"
provenance_file="${slurm_dir}/job-${SLURM_JOB_ID}-provenance.txt"
{
  printf 'started_at_utc=%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  printf 'job_id=%s\njob_name=%s\nmodel=%s\nlayout=two_tp2_workers\n' \
    "${SLURM_JOB_ID}" "${SLURM_JOB_NAME:-}" "${model_label}"
  printf 'repo=%s\ncommit=%s\npython=%s\nhf_cache=%s\n' \
    "${repo}" "${actual_commit}" "${python_bin}" "${hf_cache}"
  printf 'dataset_seal_sha256=%s\nresume_from_commits=%s\n' \
    "${dataset_seal_sha256}" "${resume_from_commits}"
  scontrol show job "${SLURM_JOB_ID}"
  module list
  nvcc --version
  "${python_bin}" -c \
    'import platform, torch, transformers, vllm; print({"python": platform.python_version(), "torch": torch.__version__, "transformers": transformers.__version__, "vllm": vllm.__version__, "cuda": torch.version.cuda})'
  printf '%s\n' "${gpu_inventory}"
} > "${provenance_file}" 2>&1

cd "${repo}"
exec 8>"${run_root}/orchestration/preparation.lock"
flock -x 8
PYTHONPATH=src "${python_bin}" \
  scripts/prepare_realistic_niah_v3_3_long_context.py \
  --run-root "${run_root}" \
  --repo-root "${repo}" \
  --expected-commit "${expected_commit}" \
  --expected-dataset-seal-sha256 "${dataset_seal_sha256}"
flock -u 8

runtime_exports="ALL,REALISTIC_NIAH_REPO_ROOT,REALISTIC_NIAH_PYTHON,REALISTIC_NIAH_HF_CACHE,REALISTIC_NIAH_EXPECTED_COMMIT,REALISTIC_NIAH_DATASET_SEAL_SHA256,REALISTIC_NIAH_RESUME_FROM_COMMITS,REALISTIC_NIAH_MODEL_LABEL,CUDA_HOME,CUDA_PATH,LD_LIBRARY_PATH,VLLM_USE_FLASHINFER_SAMPLER"
srun --exact --exclusive --nodes=1 \
  --ntasks=2 --ntasks-per-node=2 --cpus-per-task=24 \
  --gpus-per-task=2 --gpu-bind=per_task:2 --mem=480G \
  --kill-on-bad-exit=1 --label \
  --output="${worker_log_dir}/${model_label}-worker-%t-job-${SLURM_JOB_ID}.out" \
  --error="${worker_log_dir}/${model_label}-worker-%t-job-${SLURM_JOB_ID}.out" \
  --export="${runtime_exports}" \
  bash "${worker_script}" "${run_root}"

exec 9>"${run_root}/orchestration/finalize.lock"
flock -x 9
PYTHONPATH=src "${python_bin}" \
  scripts/finalize_realistic_niah_v3_3_long_context.py \
  --stimuli "${run_root}/dataset/stimuli.jsonl" \
  --run-root "${run_root}" \
  --if-ready
flock -u 9
printf 'completed_at_utc=%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" \
  >> "${provenance_file}"
printf 'V3.3 long-context model group completed: model=%s run_root=%s\n' \
  "${model_label}" "${run_root}"
