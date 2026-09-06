#!/usr/bin/env bash
set -euo pipefail

ROOT=${V6_ROOT:?set V6_ROOT to the uploaded V6 code root}
PYTHON=${V6_PYTHON:?set V6_PYTHON to the frozen V6 interpreter}
CACHE=${V6_CACHE:?set V6_CACHE to the frozen model cache}
RUN_BASE=${V6_RUN_BASE:?set V6_RUN_BASE to the completed V6 run root}
SUPERVISOR=$ROOT/scripts/supervise_realistic_niah_v6_mechanism_diagnostics.sh
QUEUE_ROOT=$RUN_BASE/mechanism_diagnostic_extension/queue_logs

for path in "$ROOT" "$PYTHON" "$CACHE" "$RUN_BASE" "$SUPERVISOR"; do
  [[ -e "$path" ]] || { echo "missing diagnostic launch input: $path" >&2; exit 4; }
done
if pgrep -af supervise_realistic_niah_v6_mechanism_diagnostics.sh \
  | grep -v "pgrep -af" >/dev/null; then
  echo "a mechanism diagnostic supervisor is already running" >&2
  pgrep -af supervise_realistic_niah_v6_mechanism_diagnostics.sh >&2
  exit 75
fi
if [[ $(nvidia-smi --query-gpu=index --format=csv,noheader | wc -l) -lt 2 ]]; then
  echo "the frozen two-worker launch requires two visible GPUs" >&2
  exit 4
fi

mkdir -p "$QUEUE_ROOT"
nohup env V6_ROOT="$ROOT" V6_PYTHON="$PYTHON" V6_CACHE="$CACHE" \
  V6_RUN_BASE="$RUN_BASE" bash "$SUPERVISOR" Qwen3-8B 0 \
  >"$QUEUE_ROOT/Qwen3-8B.log" 2>&1 &
QWEN_PID=$!
nohup env V6_ROOT="$ROOT" V6_PYTHON="$PYTHON" V6_CACHE="$CACHE" \
  V6_RUN_BASE="$RUN_BASE" bash "$SUPERVISOR" Gemma4-E4B 1 \
  >"$QUEUE_ROOT/Gemma4-E4B.log" 2>&1 &
GEMMA_PID=$!

sleep 2
kill -0 "$QWEN_PID"
kill -0 "$GEMMA_PID"
printf 'Qwen3-8B pid=%s gpu=0 log=%s\n' "$QWEN_PID" "$QUEUE_ROOT/Qwen3-8B.log"
printf 'Gemma4-E4B pid=%s gpu=1 log=%s\n' "$GEMMA_PID" "$QUEUE_ROOT/Gemma4-E4B.log"
