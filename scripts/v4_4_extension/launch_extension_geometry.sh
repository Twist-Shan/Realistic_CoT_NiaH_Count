#!/usr/bin/env bash
set -euo pipefail

ROOT=/lambda/nfs/CoT-Non-thinking-v4/runs/v4_4_nonthinking_extension_20260806
PY=/lambda/nfs/CoT-Non-thinking-v4/venv/bin/python
MAN=/lambda/nfs/CoT-Non-thinking-v4/runs/v4_4_counter_channel_20260806/packed/layer_manifest.json
SCRIPT="$ROOT/code_snapshot/analyze_v44_extension_geometry.py"
mkdir -p "$ROOT/analysis/geometry" "$ROOT/logs" "$ROOT/code_snapshot"
cp /tmp/analyze_v44_extension_geometry.py "$SCRIPT"
if ! tmux has-session -t v44ext_geometry 2>/dev/null; then
  tmux new-session -d -s v44ext_geometry \
    "'$PY' '$SCRIPT' --manifest '$MAN' --output '$ROOT/analysis/geometry' > '$ROOT/logs/geometry.log' 2>&1"
fi
tmux list-sessions
