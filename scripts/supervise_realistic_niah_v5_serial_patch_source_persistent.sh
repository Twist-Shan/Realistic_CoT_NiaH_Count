#!/usr/bin/env bash
set -euo pipefail

MODEL="${1:?usage: $0 Qwen3-8B|Gemma4-E4B}"
case "$MODEL" in
  Qwen3-8B) PATCH_LAYER=19 ;;
  Gemma4-E4B) PATCH_LAYER=16 ;;
  *) echo "unsupported model: $MODEL" >&2; exit 2 ;;
esac

CODE_ROOT="${CODE_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
PYTHON="${PYTHON:-$CODE_ROOT/.venv/bin/python}"
CACHE_DIR="${CACHE_DIR:-$CODE_ROOT/work/hf_cache}"
RUN_ROOT="${RUN_ROOT:-$CODE_ROOT/work/v5_native_count_stream/serial_source_persistent_20d10c_20260821/$MODEL}"
RUNNER="$CODE_ROOT/scripts/run_realistic_niah_v5_count_stream.py"
ANALYZER="$CODE_ROOT/scripts/analyze_realistic_niah_v5_serial_patch_source.py"
DEV_MECHANISM="$CODE_ROOT/configs/realistic_niah_v5_native_count_stream_dev.json"
CONFIRM_MECHANISM="$CODE_ROOT/configs/realistic_niah_v5_native_count_stream_confirmation_v1.json"
V5_CONFIG="$CODE_ROOT/configs/realistic_niah_v5.json"
GENERATIONS="$CODE_ROOT/work/v5_trace_parser_v2/${MODEL}_generations_reparsed.jsonl"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export HF_HOME="$CACHE_DIR"
export TOKENIZERS_PARALLELISM=false

mkdir -p "$RUN_ROOT/logs" "$RUN_ROOT/locks" "$CACHE_DIR"
LOG="$RUN_ROOT/logs/serial_patch_source_persistent.log"
exec > >(tee -a "$LOG") 2>&1
exec 9>"$RUN_ROOT/locks/serial_patch_source_persistent.lock"
if ! flock -n 9; then
  echo "another $MODEL persistent serial-source supervisor owns the lock" >&2
  exit 75
fi

for path in "$PYTHON" "$RUNNER" "$ANALYZER" "$DEV_MECHANISM" \
  "$CONFIRM_MECHANISM" "$V5_CONFIG" "$GENERATIONS"; do
  test -s "$path"
done

echo "START model=$MODEL layer=$PATCH_LAYER gpu=$CUDA_VISIBLE_DEVICES utc=$(date -u +%FT%TZ)"
cd "$CODE_ROOT"

run_phase() {
  local phase="$1"
  local mechanism="$2"
  local seed_role="$3"
  local trials="$RUN_ROOT/serial_source_${phase}"
  local analysis="$RUN_ROOT/serial_source_${phase}_analysis"
  "$PYTHON" "$RUNNER" serial-patch-source \
    --mechanism-config "$mechanism" \
    --v5-config "$V5_CONFIG" \
    --model "$MODEL" \
    --cache-dir "$CACHE_DIR" \
    --device-map auto \
    --torch-dtype bfloat16 \
    --attention-backend sdpa \
    --generations "$GENERATIONS" \
    --seed-role "$seed_role" \
    --cohort one_to_one \
    --row-panel trace_patch \
    --layer "$PATCH_LAYER" \
    --geometry suffix8 \
    --layer-mode cumulative_clamp \
    --mask-application answer_query_and_answer_tokens \
    --max-new-tokens 16 \
    --output "$trials"
  "$PYTHON" "$ANALYZER" \
    --trials "$trials" \
    --output "$analysis" \
    --phase "$phase" \
    --bootstrap-samples 10000 \
    --random-seed 20260821
}

run_phase discovery "$DEV_MECHANISM" development
DISCOVERY_PASS="$($PYTHON -c 'import json,sys; print(str(json.load(open(sys.argv[1], encoding="utf-8"))["distributed_serial_readout_pass"]).lower())' "$RUN_ROOT/serial_source_discovery_analysis/claim_gates.json")"
if [[ "$DISCOVERY_PASS" != "true" ]]; then
  "$PYTHON" - "$MODEL" "$RUN_ROOT" <<'PY'
import datetime as dt
import json
import pathlib
import sys

model, root_raw = sys.argv[1:]
root = pathlib.Path(root_raw)
claims = json.loads(
    (root / "serial_source_discovery_analysis" / "claim_gates.json").read_text(
        encoding="utf-8"
    )
)
value = {
    "schema_version": "realistic_niah_v5_serial_source_supervisor_v1",
    "status": "DISCOVERY_GATE_FAIL",
    "model_label": model,
    "discovery_seed_count": 20,
    "confirmation_opened": False,
    "distributed_serial_readout_pass": False,
    "discovery_claim_gates": claims,
    "completed_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
}
(root / "serial_source_complete.json").write_text(
    json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8"
)
print(json.dumps(value, sort_keys=True))
PY
  echo "DISCOVERY_GATE_FAIL model=$MODEL"
  exit 0
fi

run_phase confirmation "$CONFIRM_MECHANISM" confirmation

"$PYTHON" - "$MODEL" "$RUN_ROOT" <<'PY'
import datetime as dt
import json
import pathlib
import sys

model, root_raw = sys.argv[1:]
root = pathlib.Path(root_raw)

def load(path):
    return json.loads(path.read_text(encoding="utf-8"))

discovery_audit = load(root / "serial_source_discovery_analysis" / "audit.json")
confirmation_audit = load(root / "serial_source_confirmation_analysis" / "audit.json")
confirmation_claims = load(
    root / "serial_source_confirmation_analysis" / "claim_gates.json"
)
assert discovery_audit["status"] == "PASS", discovery_audit
assert confirmation_audit["status"] == "PASS", confirmation_audit
assert discovery_audit["seed_count"] == 20, discovery_audit
assert confirmation_audit["seed_count"] == 10, confirmation_audit
for phase in ("discovery", "confirmation"):
    plan = root / f"serial_source_{phase}" / "terminal_serial_pair_plan.csv"
    header = plan.read_text(encoding="utf-8").splitlines()[0].split(",")
    assert "selection_rank" not in header, header
    assert "within_cell_index" in header, header
value = {
    "schema_version": "realistic_niah_v5_serial_source_supervisor_v1",
    "status": "PASS",
    "model_label": model,
    "discovery_seed_count": 20,
    "confirmation_seed_count": 10,
    "mask_scope": "answer_query_and_answer_tokens",
    "relative_equivalence_bound": 0.20,
    "distributed_serial_readout_pass": bool(
        confirmation_claims["distributed_serial_readout_pass"]
    ),
    "confirmation_claim_gates": confirmation_claims,
    "completed_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
}
temporary = root / ".serial_source_complete.json.tmp"
temporary.write_text(
    json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8"
)
temporary.replace(root / "serial_source_complete.json")
print(json.dumps(value, sort_keys=True))
PY

echo "PASS model=$MODEL utc=$(date -u +%FT%TZ)"
