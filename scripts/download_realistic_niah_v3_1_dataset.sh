#!/usr/bin/env bash
set -euo pipefail

[[ $# -eq 1 ]] || { echo "Usage: $0 RUN_ROOT" >&2; exit 2; }
run_root="$(readlink -f "$1")"
repo="${REALISTIC_NIAH_REPO_ROOT:-$(git rev-parse --show-toplevel)}"
python_bin="${REALISTIC_NIAH_PYTHON:-python}"
hf_bin="${REALISTIC_NIAH_HF_BIN:-hf}"
dataset_id="twistshan/realistic-niah-count-empirical-law"
dataset_revision="af28be936adf92d40971aed4fa341c92b6ecf799"
dataset_dir="${run_root}/dataset"

case "${run_root}" in
  */runs/realistic_niah_v3_1/*) ;;
  *) echo "Refusing unexpected V3.1 run root: ${run_root}" >&2; exit 2 ;;
esac
command -v "${hf_bin}" >/dev/null 2>&1 \
  || { echo "Hugging Face CLI not found: ${hf_bin}" >&2; exit 2; }
[[ -x "${python_bin}" ]] || { echo "Python is not executable: ${python_bin}" >&2; exit 2; }

mkdir -p "${dataset_dir}"
"${hf_bin}" download "${dataset_id}" \
  --type dataset \
  --revision "${dataset_revision}" \
  --local-dir "${dataset_dir}"

cd "${repo}"
PYTHONPATH=src "${python_bin}" scripts/validate_realistic_niah_v3_1_dataset.py \
  --dataset-dir "${dataset_dir}" --record-source-revision
