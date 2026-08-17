#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 4 || $# -gt 5 ]]; then
  echo "usage: $0 RUN_ROOT PYTHON MODEL_CACHE STIMULI_JSONL [MODEL_LABEL]" >&2
  exit 2
fi

run_root=$1
python_bin=$2
model_cache=$3
stimuli_jsonl=$4
code_root="${run_root}/code"
generation_root="${run_root}/runs/full/generation"
capture_root="${run_root}/runs/full/capture"
rescue_config="${code_root}/configs/realistic_niah_v5_domain_transfer_rescue.json"
if [[ $# -eq 5 ]]; then
  model_labels=("$5")
else
  model_labels=(Qwen3-8B Gemma4-E4B)
fi

cd "${code_root}"
export PYTHONPATH="${code_root}/src"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

for model_label in "${model_labels[@]}"; do
  # Existing complete shards are reused.  Only a shard censored at the old
  # 4096-token ceiling is regenerated because the rescue config is higher.
  "${python_bin}" scripts/run_niah_domain_transfer.py native-generate \
    --model "${model_label}" \
    --cache-dir "${model_cache}" \
    --config "${rescue_config}" \
    --stimuli "${stimuli_jsonl}" \
    --output "${generation_root}/${model_label}"

  # Rebuild every capture so the aggregate index and the rescued response are
  # guaranteed to describe the same generation archive.
  "${python_bin}" scripts/run_niah_domain_transfer.py native-capture \
    --model "${model_label}" \
    --cache-dir "${model_cache}" \
    --config "${rescue_config}" \
    --generations "${generation_root}/${model_label}/generations.jsonl" \
    --output "${capture_root}/${model_label}" \
    --overwrite
done

if [[ $# -eq 5 ]]; then
  touch "${run_root}/runs/full/RESCUE_${5}_COMPLETE"
else
  touch "${run_root}/runs/full/RESCUE_COMPLETE"
fi
