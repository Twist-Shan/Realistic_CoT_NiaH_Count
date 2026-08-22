#!/usr/bin/env bash
set -euo pipefail

CODE_ROOT="${CODE_ROOT:-/home/ubuntu/Realistic_CoT_NiaH_Count}"
PYTHON="${PYTHON:-$CODE_ROOT/.venv/bin/python}"
EVIDENCE="$CODE_ROOT/work/v5_native_count_stream/final_count_chain_evidence_prospective_20d10c_20260821_v1"
REPORT="$CODE_ROOT/reports/NiaH_Native_Thinking_Count_Chain_Prospective_20d10c_20260821.html"
REPORT_MANIFEST="$CODE_ROOT/reports/NiaH_Native_Thinking_Count_Chain_Prospective_20d10c_20260821.manifest.json"
AUDITOR="$CODE_ROOT/scripts/audit_realistic_niah_v5_prospective_completion.py"
OUTPUT="$EVIDENCE/prospective_completion_audit.json"
LOCK="$CODE_ROOT/work/v5_native_count_stream/prospective_completion_audit_20d10c_20260821_v1.lock"
LOG="$CODE_ROOT/prospective-count-chain-completion-audit-launch.log"

test -x "$PYTHON"
test -f "$AUDITOR"

exec 9>"$LOCK"
if ! flock -n 9; then
  echo "another prospective completion-audit supervisor owns the lock" >&2
  exit 75
fi

timestamp() { date -u +%Y-%m-%dT%H:%M:%SZ; }
echo "WAIT_PROSPECTIVE_REPORT utc=$(timestamp)" | tee -a "$LOG"
while [[ ! -s "$EVIDENCE/prospective_evidence_manifest.json" || ! -s "$REPORT" || ! -s "$REPORT_MANIFEST" ]]; do
  sleep 30
done

"$PYTHON" "$AUDITOR" \
  --evidence-root "$EVIDENCE" \
  --report "$REPORT" \
  --output "$OUTPUT" | tee -a "$LOG"
echo "PROSPECTIVE_COMPLETION_AUDIT_COMPLETE utc=$(timestamp)" | tee -a "$LOG"
