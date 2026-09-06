#!/usr/bin/env bash
set -euo pipefail
experiment_root=/lambda/nfs/CoT-Native-thinking-v5/additional_experiments/task_transfer_20260905_v2
cd "$experiment_root/repo_snapshot"
export EXPERIMENT_PYTHON=/lambda/nfs/CoT-Native-thinking-v5/venv_v6_20260828/bin/python
export MODEL_CACHE=/lambda/nfs/CoT-Native-thinking-v5/hf_cache
export GPU_INDEX=0
export CUDA_VISIBLE_DEVICES="$GPU_INDEX"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export FROZEN_INPUTS="$PWD/additional_experiments/runs/task_transfer_smoke_20260905_v2/frozen"
export EXPERIMENT_OUTPUT="$experiment_root/smoke_outputs_v2_a"
"$EXPERIMENT_PYTHON" "$experiment_root/deployment/preflight.py" \
  --repo "$PWD" --cache "$MODEL_CACHE" --output "$experiment_root/deployment/preflight_at_launch.json"
bash additional_experiments/run_gpu.sh
