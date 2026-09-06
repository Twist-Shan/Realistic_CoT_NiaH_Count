#!/usr/bin/env bash
set -euo pipefail

source /etc/profile.d/modules.sh

[[ $# -eq 1 ]] || { echo "Usage: $0 RUN_ROOT" >&2; exit 2; }
run_root="$(readlink -f "$1")"
repo="${REALISTIC_NIAH_REPO_ROOT:?Set REALISTIC_NIAH_REPO_ROOT}"
python_bin="${REALISTIC_NIAH_PYTHON:?Set REALISTIC_NIAH_PYTHON}"
hf_cache="${REALISTIC_NIAH_HF_CACHE:?Set REALISTIC_NIAH_HF_CACHE}"
expected_commit="${REALISTIC_NIAH_EXPECTED_COMMIT:?Set REALISTIC_NIAH_EXPECTED_COMMIT}"

case "${run_root}" in
  */runs/realistic_niah_v3_3_long_context/*) ;;
  *) echo "Refusing unexpected V3.3 run root: ${run_root}" >&2; exit 2 ;;
esac
[[ "${expected_commit}" =~ ^[0-9a-f]{40}$ ]] \
  || { echo "Invalid expected commit" >&2; exit 2; }
[[ -x "${python_bin}" ]] || { echo "Python is not executable" >&2; exit 2; }
[[ -d "${hf_cache}" ]] || { echo "HF cache does not exist" >&2; exit 2; }
[[ -z "$(git -C "${repo}" status --short)" ]] \
  || { echo "Dataset freeze requires a clean Git worktree" >&2; exit 2; }
actual_commit="$(git -C "${repo}" rev-parse HEAD)"
[[ "${actual_commit}" == "${expected_commit}" ]] \
  || { echo "Commit mismatch: ${actual_commit} != ${expected_commit}" >&2; exit 2; }

module --force purge
module load conda
export PATH="$(dirname -- "${python_bin}"):${PATH}"
mkdir -p "${run_root}/dataset" "${run_root}/orchestration"
cd "${repo}"

PYTHONPATH=src "${python_bin}" -m pytest -q \
  tests/test_realistic_niah_v3_3_long_context.py \
  tests/test_realistic_niah_runner.py \
  tests/test_realistic_niah_v3_1_resume.py

PYTHONPATH=src "${python_bin}" \
  scripts/freeze_realistic_niah_v3_3_long_context.py \
  --output-dir "${run_root}/dataset" \
  --haystack-dir "${repo}/data/haystacks/paul_graham" \
  --haystack-corpus-manifest \
    "${repo}/data/haystacks/paul_graham/corpus_manifest.json" \
  --cache-dir "${hf_cache}"

PYTHONPATH=src "${python_bin}" \
  scripts/audit_realistic_niah_v3_3_long_context_prompts.py \
  --dataset-dir "${run_root}/dataset" \
  --output "${run_root}/orchestration/prompt_audit.json" \
  --examples "${run_root}/orchestration/prompt_examples.jsonl" \
  --cache-dir "${hf_cache}"

printf 'V3.3_DATASET_AND_PROMPT_AUDIT_OK run_root=%s commit=%s\n' \
  "${run_root}" "${actual_commit}"
