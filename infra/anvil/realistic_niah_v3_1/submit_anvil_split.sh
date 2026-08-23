#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage: submit_anvil_split.sh RUN_ROOT [OPTIONS]

Submit two independent four-H100 V3.1 jobs. The Gemma group runs
Gemma4-31B with TP=2 plus two single-GPU workers; the general group runs four
single-GPU workers. The last successful group performs the single finalizer.

Options:
  --time LIMIT          Slurm wall time per group (default: 48:00:00)
  --account NAME        Allocation account (default: mth260088-ai)
  --partition NAME      Slurm partition (default: ai)
  --constraint NAME     Node feature (default: H100)
  --expected-commit SHA Exact Git commit authorized for this formal run
  --resume-from-commits SHAS
                        Colon-separated prior commits allowed for resume
  --split-run-id ID     Coordination ID shared by both jobs
  --dry-run             Print both resolved sbatch commands
  -h, --help            Show this help
EOF
}

[[ $# -ge 1 ]] || { usage >&2; exit 2; }
case "$1" in -h|--help) usage; exit 0 ;; esac
run_root_input="$1"
shift

time_limit="48:00:00"
account="${ANVIL_ACCOUNT:-mth260088-ai}"
partition="${ANVIL_PARTITION:-ai}"
constraint="${ANVIL_CONSTRAINT:-H100}"
expected_commit="${REALISTIC_NIAH_EXPECTED_COMMIT:-}"
resume_from_commits="${REALISTIC_NIAH_RESUME_FROM_COMMITS:-}"
split_run_id="${REALISTIC_NIAH_SPLIT_RUN_ID:-}"
dry_run=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --time|--account|--partition|--constraint|--expected-commit|--resume-from-commits|--split-run-id)
      [[ $# -ge 2 ]] || { echo "Missing value for $1" >&2; exit 2; }
      option="$1"; value="$2"; shift 2
      case "${option}" in
        --time) time_limit="${value}" ;;
        --account) account="${value}" ;;
        --partition) partition="${value}" ;;
        --constraint) constraint="${value}" ;;
        --expected-commit) expected_commit="${value}" ;;
        --resume-from-commits) resume_from_commits="${value}" ;;
        --split-run-id) split_run_id="${value}" ;;
      esac
      ;;
    --dry-run) dry_run=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown option: $1" >&2; usage >&2; exit 2 ;;
  esac
done

[[ "${expected_commit}" =~ ^[0-9a-f]{40}$ ]] \
  || { echo "--expected-commit must be an exact 40-character Git SHA" >&2; exit 2; }
[[ -z "${resume_from_commits}" \
    || "${resume_from_commits}" =~ ^[0-9a-f]{40}(:[0-9a-f]{40})*$ ]] \
  || { echo "--resume-from-commits must contain colon-separated Git SHAs" >&2; exit 2; }
if [[ -z "${split_run_id}" ]]; then
  split_run_id="split-$(date -u +%Y%m%dT%H%M%SZ)-$$"
fi
[[ "${split_run_id}" =~ ^[A-Za-z0-9_.-]+$ ]] \
  || { echo "--split-run-id contains unsafe characters" >&2; exit 2; }

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
repo="${REALISTIC_NIAH_REPO_ROOT:-$(cd -- "${script_dir}/../../.." && pwd -P)}"
if [[ -n "${REALISTIC_NIAH_PYTHON:-}" ]]; then
  python_bin="${REALISTIC_NIAH_PYTHON}"
else
  [[ -n "${PROJECT:-}" && -n "${USER:-}" ]] \
    || { echo "Set PROJECT and USER, or REALISTIC_NIAH_PYTHON" >&2; exit 2; }
  python_bin="${PROJECT}/envs/${USER}/niah-v31/bin/python"
fi
if [[ -n "${REALISTIC_NIAH_HF_CACHE:-}" ]]; then
  hf_cache="${REALISTIC_NIAH_HF_CACHE}"
else
  [[ -n "${PROJECT:-}" ]] \
    || { echo "Set PROJECT or REALISTIC_NIAH_HF_CACHE" >&2; exit 2; }
  hf_cache="${PROJECT}/hf-cache"
fi

[[ -d "${run_root_input}" ]] || { echo "Run root does not exist" >&2; exit 2; }
run_root="$(readlink -f "${run_root_input}")"
case "${run_root}" in
  */runs/realistic_niah_v3_1/*) ;;
  *) echo "RUN_ROOT must be inside runs/realistic_niah_v3_1/" >&2; exit 2 ;;
esac
for required in stimuli.jsonl manifest.json audit_report.json; do
  [[ -s "${run_root}/dataset/${required}" ]] \
    || { echo "Missing frozen dataset file: ${required}" >&2; exit 2; }
done
[[ -x "${python_bin}" ]] || { echo "Python is not executable" >&2; exit 2; }
[[ -d "${hf_cache}" ]] || { echo "HF cache does not exist" >&2; exit 2; }
[[ -z "$(git -C "${repo}" status --short)" ]] \
  || { echo "Formal V3.1 submission requires a clean Git worktree" >&2; exit 2; }
actual_commit="$(git -C "${repo}" rev-parse HEAD)"
[[ "${actual_commit}" == "${expected_commit}" ]] \
  || { echo "Formal V3.1 commit mismatch" >&2; exit 2; }
PYTHONPATH="${repo}/src" "${python_bin}" \
  "${repo}/scripts/validate_realistic_niah_v3_1_dataset.py" \
  --dataset-dir "${run_root}/dataset" >/dev/null

run_tag="$(basename "${run_root}" | sed -E 's/[^A-Za-z0-9]+/-/g; s/^-+|-+$//g' | cut -c1-16)"
slurm_log_dir="${run_root}/orchestration/slurm"
mkdir -p "${slurm_log_dir}"
job_script="${script_dir}/v3_1_split_inference.slurm"
[[ -s "${job_script}" ]] || { echo "Missing split job script" >&2; exit 2; }

common_args=(
  --parsable
  --account="${account}"
  --partition="${partition}"
  --constraint="${constraint}"
  --nodes=1
  --ntasks=1
  --cpus-per-task=48
  --gpus-per-node=4
  --mem=480G
  --time="${time_limit}"
)
common_export="HOME,USER,PATH,SHELL,REALISTIC_NIAH_REPO_ROOT=${repo},REALISTIC_NIAH_PYTHON=${python_bin},REALISTIC_NIAH_HF_CACHE=${hf_cache},REALISTIC_NIAH_EXPECTED_COMMIT=${expected_commit},REALISTIC_NIAH_RESUME_FROM_COMMITS=${resume_from_commits},REALISTIC_NIAH_SPLIT_RUN_ID=${split_run_id}"

gemma_args=("${common_args[@]}"
  --job-name="rniah-v31-gemma4-${run_tag:-run}"
  --output="${slurm_log_dir}/%x-%j.out"
  --error="${slurm_log_dir}/%x-%j.out"
  --export="${common_export},REALISTIC_NIAH_SPLIT_ROLE=gemma4"
  "${job_script}" "${run_root}")
general_args=("${common_args[@]}"
  --job-name="rniah-v31-general4-${run_tag:-run}"
  --output="${slurm_log_dir}/%x-%j.out"
  --error="${slurm_log_dir}/%x-%j.out"
  --export="${common_export},REALISTIC_NIAH_SPLIT_ROLE=general4"
  "${job_script}" "${run_root}")

printf 'Resolved split Anvil submission: two independent one-node jobs, 4 H100 each\n'
printf 'Split coordination ID: %s\n' "${split_run_id}"
if [[ "${dry_run}" -eq 1 ]]; then
  printf 'sbatch'; printf ' %q' "${gemma_args[@]}"; printf '\n'
  printf 'sbatch'; printf ' %q' "${general_args[@]}"; printf '\n'
  exit 0
fi
command -v sbatch >/dev/null 2>&1 \
  || { echo "sbatch is unavailable; run this on Anvil" >&2; exit 2; }
gemma_job_id="$(sbatch "${gemma_args[@]}")"
printf 'Submitted Gemma four-GPU group %s\n' "${gemma_job_id}"
if ! general_job_id="$(sbatch "${general_args[@]}")"; then
  echo "General group submission failed; Gemma job ${gemma_job_id} remains submitted" >&2
  exit 1
fi
printf 'Submitted general four-GPU group %s\n' "${general_job_id}"
printf 'SPLIT_JOBS gemma4=%s general4=%s split_run_id=%s\n' \
  "${gemma_job_id}" "${general_job_id}" "${split_run_id}"
