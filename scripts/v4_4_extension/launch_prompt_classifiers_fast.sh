#!/usr/bin/env bash
set -euo pipefail
ROOT=/lambda/nfs/CoT-Non-thinking-v4/runs/v4_4_nonthinking_extension_20260806
PY=/lambda/nfs/CoT-Non-thinking-v4/venv/bin/python
MAN=/lambda/nfs/CoT-Non-thinking-v4/runs/v4_4_counter_channel_20260806/packed/layer_manifest.json
cp /tmp/analyze_v44_prompt_classification_fast.py "$ROOT/code_snapshot/"
tmux kill-session -t v44ext_cls_qwen 2>/dev/null || true
tmux kill-session -t v44ext_cls_gemma 2>/dev/null || true
for model in Qwen3-8B Gemma4-E4B; do
  [[ "$model" == Qwen3-8B ]] && tag=qwen || tag=gemma
  tmux new-session -d -s "v44ext_cls_${tag}" \
    "'$PY' '$ROOT/code_snapshot/analyze_v44_prompt_classification_fast.py' --manifest '$MAN' --output '$ROOT/analysis/classification_prompt_$tag' --model '$model' > '$ROOT/logs/classification_prompt_$tag.log' 2>&1"
done
tmux list-sessions
