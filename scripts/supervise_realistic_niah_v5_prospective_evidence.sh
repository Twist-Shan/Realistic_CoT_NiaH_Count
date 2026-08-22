#!/usr/bin/env bash
set -euo pipefail

CODE_ROOT="${CODE_ROOT:-/home/ubuntu/Realistic_CoT_NiaH_Count}"
PYTHON="${PYTHON:-$CODE_ROOT/.venv/bin/python}"
HISTORICAL_EVIDENCE="${HISTORICAL_EVIDENCE:-$CODE_ROOT/work/v5_native_count_stream/final_count_chain_evidence_historical_20d10c_20260821}"
OUTPUT="${OUTPUT:-$CODE_ROOT/work/v5_native_count_stream/final_count_chain_evidence_prospective_20d10c_20260821_v1}"
REPORT="${REPORT:-$CODE_ROOT/reports/NiaH_Native_Thinking_Count_Chain_Prospective_20d10c_20260821.html}"
QWEN_PROTOCOL="$CODE_ROOT/configs/realistic_niah_v5_qwen_k128_count_chain_extension_v1.json"
GEMMA_PROTOCOL="$CODE_ROOT/configs/realistic_niah_v5_gemma_k6_count_chain_extension_v1.json"
QWEN_EXTENSION="$CODE_ROOT/work/v5_native_count_stream/targeted_count_chain_k128_20d10c_20260821_v1/Qwen3-8B/prospective_extension_complete.json"
GEMMA_EXTENSION="$CODE_ROOT/work/v5_native_count_stream/targeted_count_chain_k6_20d10c_20260821_v1/Gemma4-E4B/prospective_extension_complete.json"
ASSEMBLER="$CODE_ROOT/scripts/assemble_realistic_niah_v5_prospective_count_chain_evidence.py"
REPORT_BUILDER="$CODE_ROOT/scripts/build_v5_native_count_chain_report.py"
COMPLETION_AUDITOR="$CODE_ROOT/scripts/audit_realistic_niah_v5_prospective_completion.py"
COMPLETION_AUDIT="$OUTPUT/prospective_completion_audit.json"
LOCK="$CODE_ROOT/work/v5_native_count_stream/prospective_evidence_20d10c_20260821_v1.lock"
LOG="$CODE_ROOT/prospective-count-chain-evidence-launch.log"

test -x "$PYTHON"
test -f "$QWEN_PROTOCOL"
test -f "$GEMMA_PROTOCOL"
test -f "$ASSEMBLER"
test -f "$REPORT_BUILDER"
test -f "$COMPLETION_AUDITOR"
test -f "$HISTORICAL_EVIDENCE/evidence_manifest.json"

exec 9>"$LOCK"
if ! flock -n 9; then
  echo "another prospective evidence supervisor owns the lock" >&2
  exit 75
fi

timestamp() { date -u +%Y-%m-%dT%H:%M:%SZ; }
echo "WAIT_EXTENSIONS utc=$(timestamp)" | tee -a "$LOG"
while [[ ! -s "$QWEN_EXTENSION" || ! -s "$GEMMA_EXTENSION" ]]; do
  sleep 30
done

if [[ ! -s "$OUTPUT/prospective_evidence_manifest.json" ]]; then
  "$PYTHON" "$ASSEMBLER" \
    --repo-root "$CODE_ROOT" \
    --historical-evidence-root "$HISTORICAL_EVIDENCE" \
    --extension-protocol "$QWEN_PROTOCOL" \
    --extension-protocol "$GEMMA_PROTOCOL" \
    --output "$OUTPUT" | tee -a "$LOG"
fi

"$PYTHON" "$REPORT_BUILDER" \
  --evidence-root "$OUTPUT" \
  --output "$REPORT" | tee -a "$LOG"
"$PYTHON" "$COMPLETION_AUDITOR" \
  --evidence-root "$OUTPUT" \
  --report "$REPORT" \
  --output "$COMPLETION_AUDIT" | tee -a "$LOG"
echo "PROSPECTIVE_EVIDENCE_COMPLETE utc=$(timestamp)" | tee -a "$LOG"
