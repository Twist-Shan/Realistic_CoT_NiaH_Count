#!/usr/bin/env bash
set -euo pipefail

MODEL="${1:?usage: $0 MODEL GPU PROTOCOL}"
GPU="${2:?usage: $0 MODEL GPU PROTOCOL}"
PROTOCOL="${3:?usage: $0 MODEL GPU PROTOCOL}"
CODE_ROOT="${CODE_ROOT:-/home/ubuntu/Realistic_CoT_NiaH_Count}"
PYTHON="${PYTHON:-$CODE_ROOT/.venv/bin/python}"
GATE_AUDITOR="$CODE_ROOT/scripts/audit_realistic_niah_v5_prospective_endpoint_gate.py"
FINALIZER="$CODE_ROOT/scripts/finalize_realistic_niah_v5_bank_extension.py"
BRIDGE_SUPERVISOR="${BRIDGE_SUPERVISOR:-$CODE_ROOT/scripts/supervise_realistic_niah_v5_integrated_mediator_restoration_geometry_v5.sh}"

if [[ "$MODEL" != "Qwen3-8B" && "$MODEL" != "Gemma4-E4B" ]]; then
  echo "unsupported model: $MODEL" >&2
  exit 2
fi
if [[ "$GPU" != "0" && "$GPU" != "1" ]]; then
  echo "unsupported GPU: $GPU" >&2
  exit 2
fi
[[ "$PROTOCOL" = /* ]] || PROTOCOL="$CODE_ROOT/$PROTOCOL"
test -x "$PYTHON"
test -f "$PROTOCOL"
test -f "$GATE_AUDITOR"
test -f "$FINALIZER"
test -f "$BRIDGE_SUPERVISOR"

readarray -t ROOTS < <("$PYTHON" - "$PROTOCOL" "$MODEL" <<'PY'
import json
import pathlib
import sys

protocol = json.loads(pathlib.Path(sys.argv[1]).read_text(encoding="utf-8"))
assert protocol["model_label"] == sys.argv[2]
assert [stage["name"] for stage in protocol["stages"]] == [
    "targeted_retrieval_to_final_count",
    "targeted_retrieval_to_terminal_state_to_readout",
]
for stage in protocol["stages"]:
    path = pathlib.Path(stage["root"])
    print(path if path.is_absolute() else pathlib.Path("/home/ubuntu/Realistic_CoT_NiaH_Count") / path)
PY
)
ENDPOINT_ROOT="${ROOTS[0]}"
BRIDGE_ROOT="${ROOTS[1]}"
ENDPOINT_COMPLETE="$ENDPOINT_ROOT/targeted_count_complete.json"
GATE="$ENDPOINT_ROOT/prospective_bridge_gate.json"
EXTENSION_COMPLETE="$ENDPOINT_ROOT/prospective_extension_complete.json"
LOG="$ENDPOINT_ROOT/logs/prospective_bank_extension_gatekeeper.log"

mkdir -p "$ENDPOINT_ROOT/logs" "$ENDPOINT_ROOT/locks"
exec 9>"$ENDPOINT_ROOT/locks/prospective_bank_extension_gatekeeper.lock"
if ! flock -n 9; then
  echo "another prospective extension gatekeeper owns $MODEL" >&2
  exit 75
fi

timestamp() { date -u +%Y-%m-%dT%H:%M:%SZ; }
echo "WAIT_ENDPOINT model=$MODEL gpu=$GPU utc=$(timestamp)" | tee -a "$LOG"
while [[ ! -s "$ENDPOINT_COMPLETE" ]]; do
  if ! pgrep -af "supervise_realistic_niah_v5_targeted_count_endpoint.sh $MODEL" >/dev/null; then
    echo "ENDPOINT_EXITED_WITHOUT_TERMINAL model=$MODEL utc=$(timestamp)" | tee -a "$LOG" >&2
    exit 4
  fi
  sleep 30
done

"$PYTHON" "$GATE_AUDITOR" \
  --root "$CODE_ROOT" \
  --protocol "$PROTOCOL" \
  --output "$GATE" | tee -a "$LOG"
GATE_STATUS="$($PYTHON - "$GATE" <<'PY'
import json
import pathlib
import sys
print(json.loads(pathlib.Path(sys.argv[1]).read_text(encoding="utf-8"))["status"])
PY
)"

if [[ "$GATE_STATUS" != "BRIDGE_ELIGIBLE" ]]; then
  "$PYTHON" "$FINALIZER" \
    --root "$CODE_ROOT" \
    --protocol "$PROTOCOL" \
    --output "$EXTENSION_COMPLETE" | tee -a "$LOG"
  echo "PROTOCOL_EXHAUSTED_BEFORE_BRIDGE model=$MODEL utc=$(timestamp)" | tee -a "$LOG"
  exit 0
fi

if [[ ! -s "$BRIDGE_ROOT/restoration_complete.json" ]]; then
  echo "START_SAME_BANK_FULLSPAN_BRIDGE model=$MODEL gpu=$GPU utc=$(timestamp)" | tee -a "$LOG"
  CUDA_VISIBLE_DEVICES="$GPU" \
  RUN_ROOT="$BRIDGE_ROOT" \
  TARGETED_ROOT="$ENDPOINT_ROOT" \
  GEOMETRY="full_span" \
    bash "$BRIDGE_SUPERVISOR" "$MODEL" | tee -a "$LOG"
fi

"$PYTHON" "$FINALIZER" \
  --root "$CODE_ROOT" \
  --protocol "$PROTOCOL" \
  --output "$EXTENSION_COMPLETE" | tee -a "$LOG"
echo "PROSPECTIVE_EXTENSION_TERMINAL model=$MODEL utc=$(timestamp)" | tee -a "$LOG"
