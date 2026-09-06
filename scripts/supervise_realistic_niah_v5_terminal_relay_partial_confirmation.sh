#!/usr/bin/env bash
set -euo pipefail

MODEL="${1:?usage: $0 Qwen3-8B|Gemma4-E4B}"
case "$MODEL" in
  Qwen3-8B) SOURCE_LAYER=19; RELAY_LAYER=26 ;;
  Gemma4-E4B) SOURCE_LAYER=16; RELAY_LAYER=34 ;;
  *) echo "unsupported model: $MODEL" >&2; exit 2 ;;
esac

CODE_ROOT="${CODE_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
PYTHON="${PYTHON:-$CODE_ROOT/.venv/bin/python}"
CACHE_DIR="${CACHE_DIR:-$CODE_ROOT/work/hf_cache}"
RUN_ROOT="${RUN_ROOT:-$CODE_ROOT/work/v5_native_count_stream/terminal_relay_partial_confirmation_v1/$MODEL}"
RUNNER="$CODE_ROOT/scripts/run_realistic_niah_v5_count_stream.py"
ANALYZER="$CODE_ROOT/scripts/analyze_realistic_niah_v5_terminal_relay_mediation.py"
PLAN="$CODE_ROOT/configs/realistic_niah_v5_terminal_relay_partial_mediation_confirmation_v1.json"
MECHANISM="$CODE_ROOT/configs/realistic_niah_v5_native_count_stream_confirmation_v1.json"
V5_CONFIG="$CODE_ROOT/configs/realistic_niah_v5.json"
GENERATIONS="$CODE_ROOT/work/v5_trace_parser_v2/${MODEL}_generations_reparsed.jsonl"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export HF_HOME="$CACHE_DIR"
export TOKENIZERS_PARALLELISM=false
mkdir -p "$RUN_ROOT/logs" "$RUN_ROOT/locks" "$CACHE_DIR"
LOG="$RUN_ROOT/logs/partial_confirmation.log"
exec > >(tee -a "$LOG") 2>&1
exec 9>"$RUN_ROOT/locks/partial_confirmation.lock"
if ! flock -n 9; then
  echo "another $MODEL partial-confirmation supervisor owns the lock" >&2
  exit 75
fi

for path in "$PYTHON" "$RUNNER" "$ANALYZER" "$PLAN" "$MECHANISM" \
  "$V5_CONFIG" "$GENERATIONS"; do
  test -s "$path"
done

TRIALS="$RUN_ROOT/relay_confirmation"
ANALYSIS="$RUN_ROOT/relay_analysis_confirmation"
echo "START model=$MODEL source=L$SOURCE_LAYER relay=L$RELAY_LAYER gpu=$CUDA_VISIBLE_DEVICES utc=$(date -u +%FT%TZ)"
cd "$CODE_ROOT"

"$PYTHON" "$RUNNER" terminal-relay-mediation \
  --mechanism-config "$MECHANISM" \
  --v5-config "$V5_CONFIG" \
  --model "$MODEL" \
  --cache-dir "$CACHE_DIR" \
  --device-map auto \
  --torch-dtype bfloat16 \
  --attention-backend sdpa \
  --generations "$GENERATIONS" \
  --seed-role confirmation \
  --cohort one_to_one \
  --row-panel trace_patch \
  --source-layer "$SOURCE_LAYER" \
  --relay-layer "$RELAY_LAYER" \
  --geometry suffix8 \
  --max-new-tokens 16 \
  --output "$TRIALS"

"$PYTHON" "$ANALYZER" \
  --trials "$TRIALS" \
  --output "$ANALYSIS" \
  --phase confirmation \
  --bootstrap-samples 10000 \
  --random-seed 20260821

"$PYTHON" - "$MODEL" "$RUN_ROOT" "$PLAN" <<'PY'
import datetime as dt
import hashlib
import json
import pathlib
import sys

model, root_raw, plan_raw = sys.argv[1:]
root = pathlib.Path(root_raw)
plan_path = pathlib.Path(plan_raw)
claims = json.loads(
    (root / "relay_analysis_confirmation" / "claim_gates.json").read_text(
        encoding="utf-8"
    )
)
gates = claims["gates"]
primary_ids = (
    "terminal_state_patch_effect",
    "post_terminal_suffix_specific_mediation",
)
primary_pass = all(bool(gates[name]["pass"]) for name in primary_ids)
query_pass = bool(gates["answer_query_only_mediation"]["pass"])
value = {
    "schema_version": "realistic_niah_v5_terminal_relay_partial_confirmation_v1",
    "status": "PASS" if primary_pass else "NEGATIVE",
    "model_label": model,
    "confirmation_seed_count": 10,
    "primary_gate_ids": list(primary_ids),
    "partial_mediation_pass": primary_pass,
    "answer_query_only_secondary_pass": query_pass,
    "complete_mediation_not_claimed": True,
    "greedy_control_not_required": True,
    "plan": str(plan_path.resolve()),
    "plan_sha256": hashlib.sha256(plan_path.read_bytes()).hexdigest(),
    "confirmation_claim_gates": claims,
    "completed_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
}
(root / "partial_confirmation_complete.json").write_text(
    json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8"
)
print(json.dumps(value, sort_keys=True))
PY

echo "COMPLETE model=$MODEL utc=$(date -u +%FT%TZ)"
