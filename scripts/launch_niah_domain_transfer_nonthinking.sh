#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 4 ]]; then
  echo "usage: $0 RUN_ROOT PYTHON MODEL_CACHE STIMULI_JSONL" >&2
  exit 2
fi

run_root=$1
python_bin=$2
model_cache=$3
stimuli_jsonl=$4
code_root="${run_root}/code"
output_root="${run_root}/runs/full/nonthinking"

mkdir -p "${output_root}"
cd "${code_root}"
export PYTHONPATH="${code_root}/src"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

for model_label in Qwen3-8B Gemma4-E4B; do
  "${python_bin}" scripts/run_niah_domain_transfer.py nonthinking-capture \
    --model "${model_label}" \
    --cache-dir "${model_cache}" \
    --stimuli "${stimuli_jsonl}" \
    --output "${output_root}/${model_label}"
done

touch "${output_root}/COMPLETE"
