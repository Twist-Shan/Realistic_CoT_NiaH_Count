#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage: submit_anvil_mixed.sh RUN_ROOT [OPTIONS]

Submit one eight-H100 V3.1 job. Gemma4-31B uses TP=2 while six one-GPU
workers process the other models; Gemma's GPUs join the one-GPU pool when it
finishes.

Options:
  --time LIMIT          Slurm wall time (default: 48:00:00)
  --account NAME        Allocation account (default: mth260088-ai)
  --partition NAME      Slurm partition (default: ai)
  --constraint NAME     Node feature (default: H100)
  --expected-commit SHA Exact Git commit authorized for this formal run
  --dependency SPEC     Optional afterany:JOBID or afterok:JOBID dependency
  --dry-run             Print the resolved sbatch command without submitting
  -h, --help            Show this help

Environment overrides:
  REALISTIC_NIAH_REPO_ROOT, REALISTIC_NIAH_PYTHON,
  REALISTIC_NIAH_HF_CACHE, REALISTIC_NIAH_EXPECTED_COMMIT
EOF
}

[[ $# -ge 1 ]] || { usage >&2; exit 2; }
case "$1" in
  -h|--help) usage; exit 0 ;;
esac
run_root_input="$1"
shift

time_limit="48:00:00"
account="${ANVIL_ACCOUNT:-mth260088-ai}"
partition="${ANVIL_PARTITION:-ai}"
constraint="${ANVIL_CONSTRAINT:-H100}"
expected_commit="${REALISTIC_NIAH_EXPECTED_COMMIT:-}"
dependency=""
dry_run=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --time|--account|--partition|--constraint|--expected-commit|--dependency)
      [[ $# -ge 2 ]] || { echo "Missing value for $1" >&2; exit 2; }
      option="$1"
      value="$2"
      shift 2
      case "${option}" in
        --time) time_limit="${value}" ;;
        --account) account="${value}" ;;
        --partition) partition="${value}" ;;
        --constraint) constraint="${value}" ;;
        --expected-commit) expected_commit="${value}" ;;
        --dependency) dependency="${value}" ;;
      esac
      ;;
    --dry-run) dry_run=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown option: $1" >&2; usage >&2; exit 2 ;;
  esac
done

[[ "${expected_commit}" =~ ^[0-9a-f]{40}$ ]] \
  || { echo "--expected-commit must be an exact 40-character Git SHA" >&2; exit 2; }
[[ -z "${dependency}" || "${dependency}" =~ ^(afterany|afterok):[0-9]+$ ]] \
  || { echo "--dependency must be afterany:JOBID or afterok:JOBID" >&2; exit 2; }

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
job_name="rniah-v31-mixed-${run_tag:-run}"
slurm_log_dir="${run_root}/orchestration/slurm"
mkdir -p "${slurm_log_dir}"
job_script="${script_dir}/v3_1_mixed_inference.slurm"
[[ -s "${job_script}" ]] || { echo "Missing mixed job script" >&2; exit 2; }

sbatch_args=(
  --parsable
  --account="${account}"
  --partition="${partition}"
  --constraint="${constraint}"
  --nodes=2
  --ntasks=2
  --ntasks-per-node=1
  --cpus-per-task=48
  --gpus-per-node=4
  --mem=480G
  --time="${time_limit}"
  --job-name="${job_name}"
  --output="${slurm_log_dir}/%x-%j.out"
  --error="${slurm_log_dir}/%x-%j.out"
  --export="HOME,USER,PATH,SHELL,REALISTIC_NIAH_REPO_ROOT=${repo},REALISTIC_NIAH_PYTHON=${python_bin},REALISTIC_NIAH_HF_CACHE=${hf_cache},REALISTIC_NIAH_EXPECTED_COMMIT=${expected_commit}"
)
[[ -z "${dependency}" ]] || sbatch_args+=(--dependency="${dependency}")
sbatch_args+=("${job_script}" "${run_root}")

printf 'Resolved mixed Anvil submission: 2 nodes, 8 H100, Gemma TP=2 + 6 single-GPU workers\n'
if [[ "${dry_run}" -eq 1 ]]; then
  printf 'sbatch'
  printf ' %q' "${sbatch_args[@]}"
  printf '\n'
  exit 0
fi
command -v sbatch >/dev/null 2>&1 \
  || { echo "sbatch is unavailable; run this on Anvil" >&2; exit 2; }
job_id="$(sbatch "${sbatch_args[@]}")"
printf 'Submitted mixed Slurm job %s\n' "${job_id}"
