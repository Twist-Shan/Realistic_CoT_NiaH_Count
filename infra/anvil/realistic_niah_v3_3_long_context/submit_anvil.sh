#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage: submit_anvil.sh RUN_ROOT [OPTIONS]

Submit the Gemma4-31B and Qwen3-32B V3.3 long-context holdout. Each model is an
independent job with two TP=2 workers on one four-H100 80GB node.

Required:
  --expected-commit SHA              Exact 40-character Git commit
  --dataset-seal-sha256 SHA          Exact 64-character dataset seal SHA256

Options:
  --time LIMIT                       Slurm wall time (default: 36:00:00)
  --account NAME                     Allocation (default: mth260088-ai)
  --partition NAME                   Partition (default: ai)
  --constraint NAME                  Node feature (default: H100)
  --resume-from-commits SHAS         Colon-separated prior commits allowed
  --model LABEL                      Submit only Gemma4-31B or Qwen3-32B
  --dry-run                          Print the resolved sbatch command
  -h, --help                         Show this help
EOF
}

[[ $# -ge 1 ]] || { usage >&2; exit 2; }
case "$1" in -h|--help) usage; exit 0 ;; esac
run_root_input="$1"
shift

time_limit="36:00:00"
account="${ANVIL_ACCOUNT:-mth260088-ai}"
partition="${ANVIL_PARTITION:-ai}"
constraint="${ANVIL_CONSTRAINT:-H100}"
expected_commit="${REALISTIC_NIAH_EXPECTED_COMMIT:-}"
dataset_seal_sha256="${REALISTIC_NIAH_DATASET_SEAL_SHA256:-}"
resume_from_commits="${REALISTIC_NIAH_RESUME_FROM_COMMITS:-}"
dry_run=0
selected_model=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --time|--account|--partition|--constraint|--expected-commit|--dataset-seal-sha256|--resume-from-commits|--model)
      [[ $# -ge 2 ]] || { echo "Missing value for $1" >&2; exit 2; }
      option="$1"; value="$2"; shift 2
      case "${option}" in
        --time) time_limit="${value}" ;;
        --account) account="${value}" ;;
        --partition) partition="${value}" ;;
        --constraint) constraint="${value}" ;;
        --expected-commit) expected_commit="${value}" ;;
        --dataset-seal-sha256) dataset_seal_sha256="${value}" ;;
        --resume-from-commits) resume_from_commits="${value}" ;;
        --model) selected_model="${value}" ;;
      esac
      ;;
    --dry-run) dry_run=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown option: $1" >&2; usage >&2; exit 2 ;;
  esac
done

[[ "${expected_commit}" =~ ^[0-9a-f]{40}$ ]] \
  || { echo "--expected-commit must be a full lowercase Git SHA" >&2; exit 2; }
[[ "${dataset_seal_sha256}" =~ ^[0-9a-f]{64}$ ]] \
  || { echo "--dataset-seal-sha256 must be an exact SHA256" >&2; exit 2; }
[[ -z "${resume_from_commits}" \
    || "${resume_from_commits}" =~ ^[0-9a-f]{40}(:[0-9a-f]{40})*$ ]] \
  || { echo "--resume-from-commits must contain colon-separated Git SHAs" >&2; exit 2; }
[[ -z "${selected_model}" || "${selected_model}" == "Gemma4-31B" \
    || "${selected_model}" == "Qwen3-32B" ]] \
  || { echo "--model must be Gemma4-31B or Qwen3-32B" >&2; exit 2; }

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
  */runs/realistic_niah_v3_3_long_context/*) ;;
  *) echo "RUN_ROOT must be inside runs/realistic_niah_v3_3_long_context/" >&2; exit 2 ;;
esac
for required in stimuli.jsonl manifest.json audit_report.json dataset_seal.json; do
  [[ -s "${run_root}/dataset/${required}" ]] \
    || { echo "Missing frozen dataset file: ${required}" >&2; exit 2; }
done
[[ -x "${python_bin}" ]] || { echo "Python is not executable" >&2; exit 2; }
[[ -d "${hf_cache}" ]] || { echo "HF cache does not exist" >&2; exit 2; }
[[ -z "$(git -C "${repo}" status --short)" ]] \
  || { echo "Formal submission requires a clean Git worktree" >&2; exit 2; }
actual_commit="$(git -C "${repo}" rev-parse HEAD)"
[[ "${actual_commit}" == "${expected_commit}" ]] \
  || { echo "Formal commit mismatch" >&2; exit 2; }
PYTHONPATH="${repo}/src" "${python_bin}" \
  "${repo}/scripts/validate_realistic_niah_v3_3_long_context_dataset.py" \
  --dataset-dir "${run_root}/dataset" \
  --expected-seal-sha256 "${dataset_seal_sha256}" >/dev/null

run_tag="$(basename "${run_root}" | sed -E 's/[^A-Za-z0-9]+/-/g; s/^-+|-+$//g' | cut -c1-16)"
slurm_log_dir="${run_root}/orchestration/slurm"
mkdir -p "${slurm_log_dir}"
job_script="${script_dir}/v3_3_long_context_inference.slurm"
[[ -s "${job_script}" ]] || { echo "Missing Slurm job script" >&2; exit 2; }

common_export="HOME,USER,PATH,SHELL,REALISTIC_NIAH_REPO_ROOT=${repo},REALISTIC_NIAH_PYTHON=${python_bin},REALISTIC_NIAH_HF_CACHE=${hf_cache},REALISTIC_NIAH_EXPECTED_COMMIT=${expected_commit},REALISTIC_NIAH_DATASET_SEAL_SHA256=${dataset_seal_sha256},REALISTIC_NIAH_RESUME_FROM_COMMITS=${resume_from_commits}"
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

models=("Gemma4-31B" "Qwen3-32B")
if [[ -n "${selected_model}" ]]; then
  models=("${selected_model}")
fi
printf 'Resolved V3.3 long-context submission: %s independent model job(s)\n' "${#models[@]}"
if [[ "${dry_run}" -eq 0 ]]; then
  command -v sbatch >/dev/null 2>&1 \
    || { echo "sbatch is unavailable; run this on Anvil" >&2; exit 2; }
fi
for model_label in "${models[@]}"; do
  case "${model_label}" in
    Gemma4-31B) model_tag="gemma4-31b" ;;
    Qwen3-32B) model_tag="qwen3-32b" ;;
  esac
  sbatch_args=("${common_args[@]}"
    --job-name="rniah-v33-${model_tag}-${run_tag:-run}"
    --output="${slurm_log_dir}/%x-%j.out"
    --error="${slurm_log_dir}/%x-%j.out"
    --export="${common_export},REALISTIC_NIAH_MODEL_LABEL=${model_label}"
    "${job_script}" "${run_root}")
  if [[ "${dry_run}" -eq 1 ]]; then
    printf 'sbatch'; printf ' %q' "${sbatch_args[@]}"; printf '\n'
  else
    job_id="$(sbatch "${sbatch_args[@]}")"
    printf 'Submitted %s V3.3 long-context job %s\n' \
      "${model_label}" "${job_id}"
  fi
done
