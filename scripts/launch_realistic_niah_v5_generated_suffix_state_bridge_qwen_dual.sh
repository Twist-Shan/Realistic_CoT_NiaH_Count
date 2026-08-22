#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$ROOT_DIR"
SUPERVISOR="$ROOT_DIR/scripts/supervise_realistic_niah_v5_generated_suffix_state_bridge_qwen.sh"
LAUNCH_ROOT="$ROOT_DIR/work/v5_native_count_stream/qwen_generated_suffix_state_balanced_20d10c_20260821_v1/launch"
mkdir -p "$LAUNCH_ROOT"

nohup bash "$SUPERVISOR" \
  0 count2_6_balanced terminal_span,generated_suffix_span,terminal_prefix_span \
  > "$LAUNCH_ROOT/qwen-count2-6-geometries-v1.log" 2>&1 < /dev/null &
echo "$!" > "$LAUNCH_ROOT/qwen-count2-6-geometries-v1.pid"

nohup bash "$SUPERVISOR" \
  1 count9_10_balanced terminal_span \
  > "$LAUNCH_ROOT/qwen-count9-10-terminal-v1.log" 2>&1 < /dev/null &
echo "$!" > "$LAUNCH_ROOT/qwen-count9-10-terminal-v1.pid"

printf 'count2_6_pid=%s\ncount9_10_pid=%s\n' \
  "$(cat "$LAUNCH_ROOT/qwen-count2-6-geometries-v1.pid")" \
  "$(cat "$LAUNCH_ROOT/qwen-count9-10-terminal-v1.pid")"
