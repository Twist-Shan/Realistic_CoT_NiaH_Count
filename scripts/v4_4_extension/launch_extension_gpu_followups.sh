#!/usr/bin/env bash
set -euo pipefail

ROOT=/lambda/nfs/CoT-Non-thinking-v4/runs/v4_4_nonthinking_extension_20260806
BASE=/lambda/nfs/CoT-Non-thinking-v4/runs/run_20260731_v4_numeric_presentation_v3
COUNTER=/lambda/nfs/CoT-Non-thinking-v4/runs/v4_4_counter_channel_20260806
CODE="$COUNTER/code"
PY=/lambda/nfs/CoT-Non-thinking-v4/venv/bin/python
CONFIG=/lambda/nfs/CoT-Non-thinking-v4/repo/configs/realistic_niah_v4.json
STIMULI="$BASE/dataset/stimuli.jsonl"
CACHE=/lambda/nfs/CoT-Non-thinking-v4/hf-cache
export PYTHONPATH="$CODE/src"

mkdir -p "$ROOT/code_snapshot" "$ROOT/logs" "$ROOT/analysis"
for script in \
  analyze_v44_all_token_controls.py \
  run_v44_endpoint_attention_mask.py \
  analyze_v44_endpoint_attention_mask.py \
  run_v44_token_corruption.py \
  analyze_v44_token_corruption.py \
  run_v44_prompt_subspace_ablation.py \
  analyze_v44_prompt_subspace_ablation.py; do
  cp "/tmp/$script" "$ROOT/code_snapshot/$script"
done

if tmux has-session -t v44ext_gpu_followup 2>/dev/null; then
  tmux list-sessions
  exit 0
fi

tmux new-session -d -s v44ext_gpu_followup "bash -lc '
  set -euo pipefail
  export PYTHONPATH=\"$CODE/src\"
  while tmux has-session -t v44ext_alltokens 2>/dev/null; do sleep 30; done
  test -f \"$ROOT/all_token_capture/capture_manifest.json\"
  \"$PY\" \"$ROOT/code_snapshot/analyze_v44_all_token_controls.py\" \
    --capture \"$ROOT/all_token_capture\" --packed-root \"$COUNTER/packed\" \
    --output \"$ROOT/analysis/all_token\" > \"$ROOT/logs/all_token_analysis.log\" 2>&1
  \"$PY\" \"$ROOT/code_snapshot/run_v44_endpoint_attention_mask.py\" \
    --v4-config \"$CONFIG\" --stimuli \"$STIMULI\" \
    --output \"$ROOT/endpoint_attention_mask\" --cache-dir \"$CACHE\" \
    > \"$ROOT/logs/endpoint_attention_mask.log\" 2>&1
  \"$PY\" \"$ROOT/code_snapshot/analyze_v44_endpoint_attention_mask.py\" \
    --capture \"$ROOT/endpoint_attention_mask\" --packed-root \"$COUNTER/packed\" \
    --base-run \"$BASE\" --output \"$ROOT/analysis/endpoint_attention_mask\" \
    > \"$ROOT/logs/endpoint_attention_mask_analysis.log\" 2>&1
  \"$PY\" \"$ROOT/code_snapshot/run_v44_token_corruption.py\" \
    --v4-config \"$CONFIG\" --stimuli \"$STIMULI\" \
    --output \"$ROOT/token_corruption\" --cache-dir \"$CACHE\" \
    > \"$ROOT/logs/token_corruption.log\" 2>&1
  \"$PY\" \"$ROOT/code_snapshot/analyze_v44_token_corruption.py\" \
    --input \"$ROOT/token_corruption\" --packed-root \"$COUNTER/packed\" \
    --output \"$ROOT/analysis/token_corruption\" \
    > \"$ROOT/logs/token_corruption_analysis.log\" 2>&1
  \"$PY\" \"$ROOT/code_snapshot/run_v44_prompt_subspace_ablation.py\" \
    --v4-config \"$CONFIG\" --stimuli \"$STIMULI\" --packed-root \"$COUNTER/packed\" \
    --output \"$ROOT/prompt_subspace_ablation\" --cache-dir \"$CACHE\" \
    > \"$ROOT/logs/prompt_subspace_ablation.log\" 2>&1
  \"$PY\" \"$ROOT/code_snapshot/analyze_v44_prompt_subspace_ablation.py\" \
    --input \"$ROOT/prompt_subspace_ablation\" --output \"$ROOT/analysis/prompt_subspace_ablation\" \
    > \"$ROOT/logs/prompt_subspace_ablation_analysis.log\" 2>&1
  date -Iseconds > \"$ROOT/extension_gpu_followups.complete\"
'"

tmux list-sessions
