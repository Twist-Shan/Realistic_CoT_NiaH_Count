#!/usr/bin/env bash
set -euo pipefail

ROOT=/lambda/nfs/CoT-Non-thinking-v4/runs/v4_4_nonthinking_extension_20260806
BASE=/lambda/nfs/CoT-Non-thinking-v4/runs/run_20260731_v4_numeric_presentation_v3
CODE=/lambda/nfs/CoT-Non-thinking-v4/runs/v4_4_counter_channel_20260806/code
PY=/lambda/nfs/CoT-Non-thinking-v4/venv/bin/python
CONFIG=/lambda/nfs/CoT-Non-thinking-v4/repo/configs/realistic_niah_v4.json
STIMULI="$BASE/dataset/stimuli.jsonl"
CACHE=/lambda/nfs/CoT-Non-thinking-v4/hf-cache

mkdir -p "$ROOT/code_snapshot" "$ROOT/all_token_capture" "$ROOT/logs"
cp /tmp/capture_v44_all_token_controls.py "$ROOT/code_snapshot/"
if ! tmux has-session -t v44ext_alltokens 2>/dev/null; then
  tmux new-session -d -s v44ext_alltokens \
    "export PYTHONPATH='$CODE/src'; '$PY' '$ROOT/code_snapshot/capture_v44_all_token_controls.py' \
      --v4-config '$CONFIG' --stimuli '$STIMULI' --output '$ROOT/all_token_capture' \
      --cache-dir '$CACHE' --device-map auto \
      > '$ROOT/logs/all_token_capture.log' 2>&1"
fi
tmux list-sessions
