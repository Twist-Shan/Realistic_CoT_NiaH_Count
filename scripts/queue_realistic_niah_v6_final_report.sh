#!/usr/bin/env bash
set -euo pipefail

ROOT=${V6_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}
PYTHON=${V6_PYTHON:-$ROOT/.venv/bin/python}
RUN_BASE=${V6_RUN_BASE:-$ROOT/work/realistic_niah_v6}
QUEUE_ROOT=$RUN_BASE/queue_logs
AUDIT_ROOT=$RUN_BASE/final_audit
OUTPUT_ROOT=$RUN_BASE/final_report
REPORT=$OUTPUT_ROOT/NiaH_V6_Index_Bullet_Replication_report.html
mkdir -p "$QUEUE_ROOT" "$OUTPUT_ROOT"
exec > >(tee -a "$QUEUE_ROOT/final_report.log") 2>&1

echo "[$(date --iso-8601=seconds)] WAIT final audit"
while [[ ! -s "$QUEUE_ROOT/final_audit.COMPLETE" ]]; do sleep 30; done
grep -qx PASS "$QUEUE_ROOT/final_audit.COMPLETE"

cd "$ROOT"
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH="$ROOT/src" \
  "$PYTHON" scripts/build_realistic_niah_v6_enumeration_report.py \
    --run-root "$RUN_BASE" \
    --completion-audit "$AUDIT_ROOT/suite_completion_audit.json" \
    --output "$REPORT"
grep -qx PASS "$OUTPUT_ROOT/NiaH_V6_Index_Bullet_Replication_report.COMPLETE"
printf 'PASS\n' >"$QUEUE_ROOT/final_report.COMPLETE"
echo "[$(date --iso-8601=seconds)] COMPLETE V6 self-contained report $REPORT"
