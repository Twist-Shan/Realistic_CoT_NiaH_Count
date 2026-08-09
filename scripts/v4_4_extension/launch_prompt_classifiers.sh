#!/usr/bin/env bash
set -euo pipefail

ROOT=/lambda/nfs/CoT-Non-thinking-v4/runs/v4_4_nonthinking_extension_20260806
REPO=/lambda/nfs/CoT-Non-thinking-v4/runs/v4_4_counter_channel_20260806/code
PY=/lambda/nfs/CoT-Non-thinking-v4/venv/bin/python
MAN=/lambda/nfs/CoT-Non-thinking-v4/runs/v4_4_counter_channel_20260806/packed/layer_manifest.json

mkdir -p "$ROOT/analysis" "$ROOT/logs" "$ROOT/code_snapshot"
cp "$REPO/scripts/analyze_realistic_niah_v4_4_answer_classification.py" \
  "$ROOT/code_snapshot/"

for model in Qwen3-8B Gemma4-E4B; do
  if [[ "$model" == Qwen3-8B ]]; then
    tag=qwen
  else
    tag=gemma
  fi
  session="v44ext_cls_${tag}"
  if ! tmux has-session -t "$session" 2>/dev/null; then
    tmux new-session -d -s "$session" \
      "cd '$REPO' && '$PY' scripts/analyze_realistic_niah_v4_4_answer_classification.py \
        --manifest '$MAN' \
        --output '$ROOT/analysis/classification_prompt_$tag' \
        --roles prompt_running \
        --models '$model' \
        --algorithms logistic_l2 ridge_classifier linear_svm nearest_centroid shrinkage_lda knn_k5_cosine \
        --prediction-algorithms logistic_l2 knn_k5_cosine \
        --folds 5 --pca-components 32 --n-jobs 6 \
        > '$ROOT/logs/classification_prompt_$tag.log' 2>&1"
  fi
done

tmux list-sessions
