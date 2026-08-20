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
generation_root="${run_root}/runs/full/generation"
capture_root="${run_root}/runs/full/capture"

mkdir -p "${generation_root}" "${capture_root}"
cd "${code_root}"
export PYTHONPATH="${code_root}/src"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

for model_label in Qwen3-8B Gemma4-E4B; do
  "${python_bin}" scripts/run_niah_domain_transfer.py native-generate \
    --model "${model_label}" \
    --cache-dir "${model_cache}" \
    --stimuli "${stimuli_jsonl}" \
    --output "${generation_root}/${model_label}"
  "${python_bin}" scripts/run_niah_domain_transfer.py native-capture \
    --model "${model_label}" \
    --cache-dir "${model_cache}" \
    --generations "${generation_root}/${model_label}/generations.jsonl" \
    --output "${capture_root}/${model_label}"
done

touch "${run_root}/runs/full/COMPLETE"
