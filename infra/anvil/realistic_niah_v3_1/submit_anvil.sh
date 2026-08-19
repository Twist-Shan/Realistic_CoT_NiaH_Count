#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage: submit_anvil.sh RUN_ROOT [OPTIONS]

Submit Realistic NIAH V3.1 inference to Purdue Anvil. The default is eight
independent one-H100 workers distributed as four tasks per H100 node.

Options:
  --workers N           Number of one-GPU workers (default: 8; maximum: 12)
  --time LIMIT          Slurm wall time (default: 48:00:00)
  --account NAME        Allocation account (default: mth260088-ai)
  --partition NAME      Slurm partition (default: ai)
  --constraint NAME     Node feature (default: H100)
  --cpus-per-worker N   CPU cores per GPU worker (default: 12)
  --mem-per-node SIZE   Memory per node (default: 120G per resident worker)
  --expected-commit SHA Exact Git commit authorized for this formal run
  --dry-run             Print the resolved sbatch command without submitting
  -h, --help            Show this help

Environment overrides:
  REALISTIC_NIAH_REPO_ROOT, REALISTIC_NIAH_PYTHON,
  REALISTIC_NIAH_HF_CACHE, ANVIL_ACCOUNT, ANVIL_PARTITION,
  ANVIL_CONSTRAINT, ANVIL_WORKERS, ANVIL_TIME_LIMIT,
  ANVIL_CPUS_PER_WORKER, ANVIL_MEM_PER_NODE,
  REALISTIC_NIAH_EXPECTED_COMMIT
EOF
}

[[ $# -ge 1 ]] || { usage >&2; exit 2; }
case "$1" in
  -h|--help) usage; exit 0 ;;
esac

run_root_input="$1"
shift
workers="${ANVIL_WORKERS:-8}"
time_limit="${ANVIL_TIME_LIMIT:-48:00:00}"
account="${ANVIL_ACCOUNT:-mth260088-ai}"
partition="${ANVIL_PARTITION:-ai}"
constraint="${ANVIL_CONSTRAINT:-H100}"
cpus_per_worker="${ANVIL_CPUS_PER_WORKER:-12}"
mem_per_node="${ANVIL_MEM_PER_NODE:-}"
expected_commit="${REALISTIC_NIAH_EXPECTED_COMMIT:-}"
dry_run=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --workers|--time|--account|--partition|--constraint|--cpus-per-worker|--mem-per-node|--expected-commit)
      [[ $# -ge 2 ]] || { echo "Missing value for $1" >&2; exit 2; }
      option="$1"
      value="$2"
      shift 2
      case "${option}" in
        --workers) workers="${value}" ;;
        --time) time_limit="${value}" ;;
        --account) account="${value}" ;;
        --partition) partition="${value}" ;;
        --constraint) constraint="${value}" ;;
        --cpus-per-worker) cpus_per_worker="${value}" ;;
        --mem-per-node) mem_per_node="${value}" ;;
        --expected-commit) expected_commit="${value}" ;;
      esac
      ;;
    --dry-run)
      dry_run=1
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown option: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

[[ "${workers}" =~ ^[1-9][0-9]*$ ]] && [[ "${workers}" -le 12 ]] \
  || { echo "--workers must be an integer from 1 through 12" >&2; exit 2; }
[[ "${cpus_per_worker}" =~ ^[1-9][0-9]*$ ]] \
  || { echo "--cpus-per-worker must be a positive integer" >&2; exit 2; }
[[ -n "${account}" && -n "${partition}" && -n "${constraint}" ]] \
  || { echo "Account, partition, and constraint must be non-empty" >&2; exit 2; }
[[ "${expected_commit}" =~ ^[0-9a-f]{40}$ ]] \
  || { echo "--expected-commit must be an exact 40-character Git SHA" >&2; exit 2; }

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

[[ -d "${run_root_input}" ]] \
  || { echo "Run root does not exist: ${run_root_input}" >&2; exit 2; }
run_root="$(readlink -f "${run_root_input}")"
case "${run_root}" in
  */runs/realistic_niah_v3_1/*) ;;
  *) echo "RUN_ROOT must be inside runs/realistic_niah_v3_1/" >&2; exit 2 ;;
esac
for required in stimuli.jsonl manifest.json audit_report.json; do
  [[ -s "${run_root}/dataset/${required}" ]] \
    || { echo "Missing frozen dataset file: ${run_root}/dataset/${required}" >&2; exit 2; }
done
[[ -d "${repo}/.git" ]] || { echo "Not a Git repository: ${repo}" >&2; exit 2; }
[[ -x "${python_bin}" ]] || { echo "Python is not executable: ${python_bin}" >&2; exit 2; }
[[ -d "${hf_cache}" ]] || { echo "HF cache does not exist: ${hf_cache}" >&2; exit 2; }
repo_status="$(git -C "${repo}" status --short)"
if [[ -n "${repo_status}" ]]; then
  echo "Formal V3.1 submission requires a clean Git worktree" >&2
  exit 2
fi
actual_commit="$(git -C "${repo}" rev-parse HEAD)"
[[ "${actual_commit}" == "${expected_commit}" ]] \
  || { echo "Formal V3.1 commit mismatch: ${actual_commit} != ${expected_commit}" >&2; exit 2; }
PYTHONPATH="${repo}/src" "${python_bin}" \
  "${repo}/scripts/validate_realistic_niah_v3_1_dataset.py" \
  --dataset-dir "${run_root}/dataset" >/dev/null

nodes="$(((workers + 3) / 4))"
tasks_per_node=4
if [[ "${workers}" -lt 4 ]]; then
  tasks_per_node="${workers}"
fi
if [[ -z "${mem_per_node}" ]]; then
  mem_per_node="$((tasks_per_node * 120))G"
fi
run_tag="$(basename "${run_root}" | sed -E 's/[^A-Za-z0-9]+/-/g; s/^-+|-+$//g' | cut -c1-20)"
run_tag="${run_tag:-run}"
job_name="rniah-v31-${run_tag}"
slurm_log_dir="${run_root}/orchestration/slurm"
mkdir -p "${slurm_log_dir}"
job_script="${script_dir}/v3_1_inference.slurm"
[[ -s "${job_script}" ]] || { echo "Missing job script: ${job_script}" >&2; exit 2; }

export REALISTIC_NIAH_REPO_ROOT="${repo}"
export REALISTIC_NIAH_PYTHON="${python_bin}"
export REALISTIC_NIAH_HF_CACHE="${hf_cache}"
export REALISTIC_NIAH_TASKS_PER_NODE="${tasks_per_node}"
export REALISTIC_NIAH_EXPECTED_COMMIT="${expected_commit}"

sbatch_args=(
  --parsable
  --account="${account}"
  --partition="${partition}"
  --constraint="${constraint}"
  --nodes="${nodes}"
  --ntasks="${workers}"
  --ntasks-per-node="${tasks_per_node}"
  --cpus-per-task="${cpus_per_worker}"
  --gpus-per-task=1
  --mem="${mem_per_node}"
  --time="${time_limit}"
  --job-name="${job_name}"
  --output="${slurm_log_dir}/%x-%j.out"
  --error="${slurm_log_dir}/%x-%j.out"
  --export=HOME,USER,PATH,SHELL,REALISTIC_NIAH_REPO_ROOT,REALISTIC_NIAH_PYTHON,REALISTIC_NIAH_HF_CACHE,REALISTIC_NIAH_TASKS_PER_NODE,REALISTIC_NIAH_EXPECTED_COMMIT
  "${job_script}"
  "${run_root}"
)

printf 'Resolved Anvil V3.1 submission: workers=%s nodes=%s tasks_per_node=%s account=%s partition=%s constraint=%s\n' \
  "${workers}" "${nodes}" "${tasks_per_node}" "${account}" "${partition}" "${constraint}"
if [[ "${dry_run}" -eq 1 ]]; then
  printf 'sbatch'
  printf ' %q' "${sbatch_args[@]}"
  printf '\n'
  exit 0
fi
command -v sbatch >/dev/null 2>&1 \
  || { echo "sbatch is unavailable; run this entry point on Anvil" >&2; exit 2; }
job_id="$(sbatch "${sbatch_args[@]}")"
printf 'Submitted Slurm job %s. Monitor with: squeue -j %q\n' "${job_id}" "${job_id}"
