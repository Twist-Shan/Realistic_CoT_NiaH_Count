#!/usr/bin/env bash
set -euo pipefail
# Run from the extracted repo_snapshot directory in a confirmed Filestream mount.
# Required variables avoid guessing the remote interpreter, checkpoint cache or GPU.
: "${EXPERIMENT_PYTHON:?Set the existing CUDA Python interpreter path}"
: "${MODEL_CACHE:?Set the existing Hugging Face model cache directory}"
: "${GPU_INDEX:?Select an idle GPU index after inspecting nvidia-smi}"
: "${FROZEN_INPUTS:?Set the frozen input directory}"
: "${EXPERIMENT_OUTPUT:?Set a NEW output directory on the confirmed Filestream mount}"
test -f "$FROZEN_INPUTS/freeze_audit.json"
test -d "$MODEL_CACHE"
test ! -e "$EXPERIMENT_OUTPUT"
used_memory=$(nvidia-smi -i "$GPU_INDEX" --query-gpu=memory.used --format=csv,noheader,nounits)
if (( used_memory > 1024 )); then
  printf 'Selected GPU has %s MiB allocated; choose an idle GPU.\n' "$used_memory" >&2
  exit 1
fi
mkdir -p "$EXPERIMENT_OUTPUT"
exec 9>"$EXPERIMENT_OUTPUT/run.lock"
flock -n 9
export CUDA_VISIBLE_DEVICES="$GPU_INDEX"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
for model in Qwen3-8B Gemma4-E4B; do
  "$EXPERIMENT_PYTHON" additional_experiments/run.py \
    --frozen "$FROZEN_INPUTS" --model "$model" --cache-dir "$MODEL_CACHE" \
    --device-map auto --output "$EXPERIMENT_OUTPUT/$model" \
    >"$EXPERIMENT_OUTPUT/${model}.log" 2>&1
done
"$EXPERIMENT_PYTHON" additional_experiments/analyze.py \
  --frozen "$FROZEN_INPUTS" \
  --outputs "$EXPERIMENT_OUTPUT/Qwen3-8B" "$EXPERIMENT_OUTPUT/Gemma4-E4B" \
  --output "$EXPERIMENT_OUTPUT/analysis" \
  >"$EXPERIMENT_OUTPUT/analysis.log" 2>&1
