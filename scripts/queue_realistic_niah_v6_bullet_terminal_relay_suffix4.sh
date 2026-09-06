#!/usr/bin/env bash
set -euo pipefail

ROOT=${V6_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}
PYTHON=${V6_PYTHON:-$ROOT/.venv/bin/python}
RUN_BASE=${V6_RUN_BASE:-$ROOT/work/realistic_niah_v6}
GPU_INDEX=${V6_BULLET_SUFFIX4_GPU:-0}
AMENDMENT=$ROOT/configs/realistic_niah_v6_bullet_terminal_relay_suffix4_amendment_v1.json
REPORT_ROOT=${V6_ANSWER_TRACE_REPORT_ROOT:-$RUN_BASE/answer_trace_extension_report}
LOG_ROOT=$RUN_BASE/queue_logs
COMPLETE=$LOG_ROOT/bullet_terminal_relay_suffix4.COMPLETE

mkdir -p "$LOG_ROOT"
exec > >(tee -a "$LOG_ROOT/bullet_terminal_relay_suffix4.log") 2>&1
exec 9>"$LOG_ROOT/bullet_terminal_relay_suffix4.lock"
if ! flock -n 9; then
  echo "another Bullet suffix4 relay queue owns the lock" >&2
  exit 75
fi

cd "$ROOT"
for model in Qwen3-8B Gemma4-E4B; do
  bash "$ROOT/scripts/supervise_realistic_niah_v6_bullet_terminal_relay_suffix4.sh" \
    "$model" "$GPU_INDEX"
done

"$PYTHON" scripts/build_realistic_niah_v6_answer_trace_extension_report.py \
  --run-root "$RUN_BASE" --output-root "$REPORT_ROOT" \
  --relay-geometry-amendment "$AMENDMENT"

printf 'PASS\n' >"$COMPLETE"
echo "[$(date --iso-8601=seconds)] COMPLETE Bullet suffix4 relay queue"
