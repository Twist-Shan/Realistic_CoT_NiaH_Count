#!/usr/bin/env bash
set -euo pipefail

[[ $# -eq 1 ]] || { echo "Usage: $0 RUN_ROOT" >&2; exit 2; }
[[ -n "${SLURM_JOB_ID:-}" && -n "${SLURM_NTASKS:-}" ]] \
  || { echo "This script must run inside a Slurm allocation" >&2; exit 2; }

run_root="$(readlink -f "$1")"
repo="${REALISTIC_NIAH_REPO_ROOT:?Set REALISTIC_NIAH_REPO_ROOT}"
python_bin="${REALISTIC_NIAH_PYTHON:?Set REALISTIC_NIAH_PYTHON}"
hf_cache="${REALISTIC_NIAH_HF_CACHE:?Set REALISTIC_NIAH_HF_CACHE}"
tasks_per_node="${REALISTIC_NIAH_TASKS_PER_NODE:-4}"
script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"

case "${run_root}" in
  */runs/realistic_niah_v3_1/*) ;;
  *) echo "Refusing unexpected V3.1 run root: ${run_root}" >&2; exit 2 ;;
esac
[[ -x "${python_bin}" ]] || { echo "Python is not executable: ${python_bin}" >&2; exit 2; }
[[ -d "${hf_cache}" ]] || { echo "HF cache does not exist: ${hf_cache}" >&2; exit 2; }
[[ -z "$(git -C "${repo}" status --short)" ]] \
  || { echo "Formal V3.1 execution requires a clean Git worktree" >&2; exit 2; }

module --force purge
module load modtree/gpu
module load conda

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
  "${python_bin}" -c \
    'import platform, torch, transformers, vllm; print({"python": platform.python_version(), "torch": torch.__version__, "transformers": transformers.__version__, "vllm": vllm.__version__, "cuda": torch.version.cuda})'
  nvidia-smi --query-gpu=name,uuid,memory.total,driver_version --format=csv
} > "${provenance_file}" 2>&1

cd "${repo}"
PYTHONPATH=src "${python_bin}" scripts/prepare_realistic_niah_v3_1.py \
  --run-root "${run_root}" --repo-root "${repo}"

srun \
  --ntasks="${SLURM_NTASKS}" \
  --ntasks-per-node="${tasks_per_node}" \
  --gpus-per-task=1 \
  --gpu-bind=single:1 \
  --cpu-bind=cores \
  --kill-on-bad-exit=1 \
  --label \
  bash "${script_dir}/run_slurm_task.sh" "${run_root}"

bash scripts/finalize_realistic_niah_v3_1.sh "${run_root}"
printf 'completed_at_utc=%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" \
  >> "${provenance_file}"
printf 'V3.1 inference and merge completed for %s\n' "${run_root}"
