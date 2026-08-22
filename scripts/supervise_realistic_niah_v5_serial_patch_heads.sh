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
RUN_ROOT="${RUN_ROOT:-$CODE_ROOT/work/v5_native_count_stream/serial_chain_20d10c_20260820/$MODEL}"
RUNNER="$CODE_ROOT/scripts/run_realistic_niah_v5_count_stream.py"
ANALYZER="$CODE_ROOT/scripts/analyze_realistic_niah_v5_serial_patch_heads.py"
DEV_MECHANISM="$CODE_ROOT/configs/realistic_niah_v5_native_count_stream_dev.json"
CONFIRM_MECHANISM="$CODE_ROOT/configs/realistic_niah_v5_native_count_stream_confirmation_v1.json"
V5_CONFIG="$CODE_ROOT/configs/realistic_niah_v5.json"
GENERATIONS="$CODE_ROOT/work/v5_trace_parser_v2/${MODEL}_generations_reparsed.jsonl"

# The previous capture itself used only canonical ranking seeds 1234..1243.
# Reusing it avoids 100 redundant long-prefix forwards; plan-broad re-audits
# exact 10-seed x 10-count coverage before any new K-selection outcome opens.
LEGACY_RANKING_CAPTURE="$CODE_ROOT/work/v5_native_count_stream/stage1_20260820/$MODEL/broad_ranking_capture"
NEW_RANKING_CAPTURE="$RUN_ROOT/broad_ranking_capture"
HEAD_PLAN_ROOT="$RUN_ROOT/broad_plan_trace"
HEAD_PLAN="$HEAD_PLAN_ROOT/answer_broad_head_plan.csv"
K_TRIALS="$RUN_ROOT/broad_k_grid_trace"
K64_TRIALS="$RUN_ROOT/broad_K64_trace"
K_SELECTION="$RUN_ROOT/k_selection_trace"
DISCOVERY_TRIALS="$RUN_ROOT/serial_factorial_discovery"
DISCOVERY_ANALYSIS="$RUN_ROOT/serial_analysis_discovery"
CONFIRM_TRIALS="$RUN_ROOT/serial_factorial_confirmation"
CONFIRM_ANALYSIS="$RUN_ROOT/serial_analysis_confirmation"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export HF_HOME="$CACHE_DIR"
export TOKENIZERS_PARALLELISM=false

mkdir -p "$RUN_ROOT/logs" "$RUN_ROOT/locks" "$CACHE_DIR"
LOG="$RUN_ROOT/logs/serial_patch_heads.log"
exec > >(tee -a "$LOG") 2>&1
exec 9>"$RUN_ROOT/locks/serial_patch_heads.lock"
if ! flock -n 9; then
  echo "another $MODEL serial-patch-head supervisor owns the lock" >&2
  exit 75
fi

for path in "$PYTHON" "$RUNNER" "$ANALYZER" "$DEV_MECHANISM" \
  "$CONFIRM_MECHANISM" "$V5_CONFIG" "$GENERATIONS"; do
  test -s "$path"
done

echo "START model=$MODEL layer=$PATCH_LAYER gpu=$CUDA_VISIBLE_DEVICES utc=$(date -u +%FT%TZ)"
cd "$CODE_ROOT"

if [[ -s "$LEGACY_RANKING_CAPTURE/manifest.json" ]]; then
  RANKING_CAPTURE="$LEGACY_RANKING_CAPTURE"
else
  RANKING_CAPTURE="$NEW_RANKING_CAPTURE"
  "$PYTHON" "$RUNNER" capture-broad \
    --mechanism-config "$DEV_MECHANISM" \
    --v5-config "$V5_CONFIG" \
    --model "$MODEL" \
    --cache-dir "$CACHE_DIR" \
    --device-map auto \
    --torch-dtype bfloat16 \
    --attention-backend sdpa \
    --generations "$GENERATIONS" \
    --seed-role development \
    --cohort parser_hit \
    --row-panel broad_ranking \
    --output "$RANKING_CAPTURE"
fi

"$PYTHON" "$RUNNER" plan-broad \
  --mechanism-config "$DEV_MECHANISM" \
  --captures "$RANKING_CAPTURE" \
  --model "$MODEL" \
  --source-group trace_items \
  --bank-sizes 1 2 4 8 16 32 64 \
  --random-seed 20260820 \
  --output "$HEAD_PLAN_ROOT"

"$PYTHON" "$RUNNER" broad-heads \
  --mechanism-config "$DEV_MECHANISM" \
  --v5-config "$V5_CONFIG" \
  --model "$MODEL" \
  --cache-dir "$CACHE_DIR" \
  --device-map auto \
  --torch-dtype bfloat16 \
  --attention-backend sdpa \
  --generations "$GENERATIONS" \
  --seed-role development \
  --cohort parser_hit \
  --row-panel broad_k_selection \
  --plan "$HEAD_PLAN" \
  --bank-sizes 1 2 4 8 16 32 \
  --skip-greedy \
  --output "$K_TRIALS"

"$PYTHON" "$RUNNER" select-broad-k \
  --mechanism-config "$DEV_MECHANISM" \
  --model "$MODEL" \
  --source-group trace_items \
  --plan "$HEAD_PLAN" \
  --trials "$K_TRIALS" \
  --random-seed 20260820 \
  --output "$K_SELECTION"

DECISION_STATUS="$($PYTHON -c 'import json,sys; print(json.load(open(sys.argv[1], encoding="utf-8"))["status"])' "$K_SELECTION/k_selection_decision.json")"
if [[ "$DECISION_STATUS" == "requires_boundary_extension" ]]; then
  "$PYTHON" "$RUNNER" broad-heads \
    --mechanism-config "$DEV_MECHANISM" \
    --v5-config "$V5_CONFIG" \
    --model "$MODEL" \
    --cache-dir "$CACHE_DIR" \
    --device-map auto \
    --torch-dtype bfloat16 \
    --attention-backend sdpa \
    --generations "$GENERATIONS" \
    --seed-role development \
    --cohort parser_hit \
    --row-panel broad_k_selection \
    --plan "$HEAD_PLAN" \
    --bank-sizes 64 \
    --skip-greedy \
    --output "$K64_TRIALS"
  "$PYTHON" "$RUNNER" select-broad-k \
    --mechanism-config "$DEV_MECHANISM" \
    --model "$MODEL" \
    --source-group trace_items \
    --plan "$HEAD_PLAN" \
    --trials "$K_TRIALS" "$K64_TRIALS" \
    --random-seed 20260820 \
    --output "$K_SELECTION"
  DECISION_STATUS="$($PYTHON -c 'import json,sys; print(json.load(open(sys.argv[1], encoding="utf-8"))["status"])' "$K_SELECTION/k_selection_decision.json")"
fi

if [[ "$DECISION_STATUS" != "frozen_for_confirmation" ]]; then
  "$PYTHON" - "$MODEL" "$RUN_ROOT" "$DECISION_STATUS" <<'PY'
import datetime as dt
import json
import pathlib
import sys

model, root_raw, decision = sys.argv[1:]
root = pathlib.Path(root_raw)
value = {
    "schema_version": "realistic_niah_v5_serial_patch_heads_supervisor_v1",
    "status": "NO_FROZEN_TRACE_BANK",
    "model_label": model,
    "k_selection_status": decision,
    "serial_readout_pass": False,
    "completed_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
}
(root / "serial_complete.json").write_text(
    json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8"
)
print(json.dumps(value, sort_keys=True))
PY
  echo "NO_FROZEN_TRACE_BANK model=$MODEL status=$DECISION_STATUS"
  exit 0
fi

FROZEN_HEAD_PLAN="$K_SELECTION/frozen_answer_broad_head_plan.csv"
test -s "$FROZEN_HEAD_PLAN"

run_factorial() {
  local phase="$1"
  local mechanism="$2"
  local seed_role="$3"
  local trials="$4"
  local analysis="$5"
  "$PYTHON" "$RUNNER" serial-patch-heads \
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
    --head-plan "$FROZEN_HEAD_PLAN" \
    --layer "$PATCH_LAYER" \
    --geometry suffix8 \
    --layer-mode cumulative_clamp \
    --max-new-tokens 16 \
    --output "$trials"
  "$PYTHON" "$ANALYZER" \
    --trials "$trials" \
    --output "$analysis" \
    --phase "$phase" \
    --bootstrap-samples 10000 \
    --random-seed 20260820
}

run_factorial discovery "$DEV_MECHANISM" development \
  "$DISCOVERY_TRIALS" "$DISCOVERY_ANALYSIS"
run_factorial confirmation "$CONFIRM_MECHANISM" confirmation \
  "$CONFIRM_TRIALS" "$CONFIRM_ANALYSIS"

"$PYTHON" - "$MODEL" "$RUN_ROOT" "$FROZEN_HEAD_PLAN" <<'PY'
import datetime as dt
import hashlib
import json
import pathlib
import sys

model, root_raw, head_plan_raw = sys.argv[1:]
root = pathlib.Path(root_raw)
head_plan = pathlib.Path(head_plan_raw)

def load(path):
    return json.loads(path.read_text(encoding="utf-8"))

discovery_audit = load(root / "serial_analysis_discovery" / "audit.json")
confirmation_audit = load(root / "serial_analysis_confirmation" / "audit.json")
confirmation_claims = load(
    root / "serial_analysis_confirmation" / "claim_gates.json"
)
assert discovery_audit["status"] == "PASS", discovery_audit
assert confirmation_audit["status"] == "PASS", confirmation_audit
assert discovery_audit["seed_count"] == 20, discovery_audit
assert confirmation_audit["seed_count"] == 10, confirmation_audit
for phase in ("discovery", "confirmation"):
    plan = root / f"serial_factorial_{phase}" / "terminal_serial_pair_plan.csv"
    header = plan.read_text(encoding="utf-8").splitlines()[0].split(",")
    assert "selection_rank" not in header, header
    assert "within_cell_index" in header, header
value = {
    "schema_version": "realistic_niah_v5_serial_patch_heads_supervisor_v1",
    "status": "PASS",
    "model_label": model,
    "completed_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
    "discovery_seed_count": 20,
    "confirmation_seed_count": 10,
    "head_plan": str(head_plan.resolve()),
    "head_plan_sha256": hashlib.sha256(head_plan.read_bytes()).hexdigest(),
    "relative_equivalence_bound": 0.20,
    "serial_readout_pass": bool(confirmation_claims["serial_readout_pass"]),
    "confirmation_claim_gates": confirmation_claims,
}
temporary = root / ".serial_complete.json.tmp"
temporary.write_text(
    json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8"
)
temporary.replace(root / "serial_complete.json")
print(json.dumps(value, sort_keys=True))
PY

echo "PASS model=$MODEL utc=$(date -u +%FT%TZ)"
