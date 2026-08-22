#!/usr/bin/env bash
set -euo pipefail

MODEL=${1:?usage: $0 MODEL GPU_INDEX}
GPU_INDEX=${2:?usage: $0 MODEL GPU_INDEX}
ROOT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$ROOT_DIR"

case "$MODEL" in
  Qwen3-8B)
    SOURCE_LAYER=19
    BANK_SIZE=128
    BANK_SHA=ef30a8a083468c6e88cb5b0924403884ad758fedbc743de36dd03ab9bc4a742b
    TARGETED_ROOT="$ROOT_DIR/work/v5_native_count_stream/targeted_count_chain_k128_20d10c_20260821_v1/$MODEL"
    ;;
  Gemma4-E4B)
    SOURCE_LAYER=16
    BANK_SIZE=6
    BANK_SHA=2a7652c68454a5333f19324ec5517fe8c22b03ef4955088a283229c8576211b1
    TARGETED_ROOT="$ROOT_DIR/work/v5_native_count_stream/targeted_count_chain_k6_20d10c_20260821_v1/$MODEL"
    ;;
  *)
    echo "unsupported model: $MODEL" >&2
    exit 2
    ;;
esac

PYTHON="$ROOT_DIR/.venv/bin/python"
RUNNER="$ROOT_DIR/scripts/run_realistic_niah_v5_targeted_counter_write.py"
ANALYZER="$ROOT_DIR/scripts/analyze_realistic_niah_v5_targeted_counter_write.py"
PROTOCOL="$ROOT_DIR/configs/realistic_niah_v5_targeted_counter_write_v1.json"
DEV_CONFIG="$ROOT_DIR/configs/realistic_niah_v5_native_count_stream_dev.json"
CONFIRM_CONFIG="$ROOT_DIR/configs/realistic_niah_v5_native_count_stream_confirmation_v1.json"
V5_CONFIG="$ROOT_DIR/configs/realistic_niah_v5.json"
GENERATIONS="$ROOT_DIR/work/v5_trace_parser_v2/${MODEL}_generations_reparsed.jsonl"
BANK_PLAN="$TARGETED_ROOT/frozen_targeted_count_plan.csv"
TARGETED_REGISTRY="$TARGETED_ROOT/final_transition_registry/selected_anchor_registry.jsonl"
GRAMMAR_ROOT="$ROOT_DIR/work/v5_native_count_stream/grammar_terminal_span_decomposition_k128_k6_20d10c_20260821_v1/$MODEL"
ANCHORS="$GRAMMAR_ROOT/frozen_grammar_span_anchor_panel.jsonl"
ANCHOR_MANIFEST="$GRAMMAR_ROOT/frozen_grammar_span_anchor_manifest.json"
UPSTREAM_COMPLETE="$ROOT_DIR/work/v5_native_count_stream/commit_state_to_targeted_query_20d10c_20260822_v1/$MODEL/commit_to_query_complete.json"
OUTPUT_ROOT="$ROOT_DIR/work/v5_native_count_stream/teacher_forced_targeted_counter_write_k128_k6_20d10c_20260822_v1/$MODEL"
COMPLETE="$OUTPUT_ROOT/targeted_counter_write_complete.json"
LOG="$OUTPUT_ROOT/logs/targeted_counter_write.log"
mkdir -p "$OUTPUT_ROOT/logs" "$OUTPUT_ROOT/locks"

exec 9>"$OUTPUT_ROOT/locks/targeted_counter_write.lock"
if ! flock -n 9; then
  echo "another $MODEL targeted-counter-write supervisor owns the lock" >&2
  exit 3
fi

"$PYTHON" - "$PROTOCOL" "$MODEL" "$BANK_PLAN" "$ANCHOR_MANIFEST" "$BANK_SIZE" "$BANK_SHA" <<'PY'
import json, pathlib, pandas as pd, sys

protocol_path, model, bank_path, anchor_manifest, bank_size, bank_sha = sys.argv[1:]
protocol = json.loads(pathlib.Path(protocol_path).read_text())
assert protocol["status"] == "FROZEN_BEFORE_ANY_TEACHER_FORCED_WRITE_OUTCOME"
assert protocol["seed_contract"]["discovery"] == list(range(1234, 1254))
assert protocol["seed_contract"]["confirmation"] == list(range(1254, 1264))
assert protocol["seed_contract"]["outcome_blind"] is True
assert protocol["seed_contract"]["selection_rank_used"] is False
assert int(protocol["models"][model]["targeted_bank_size"]) == int(bank_size)
assert protocol["models"][model]["targeted_bank_sha256"] == bank_sha
frame = pd.read_csv(bank_path)
assert "selection_rank" not in frame.columns
selected = frame.loc[
    frame["model_label"].astype(str).eq(model)
    & frame["condition"].astype(str).eq("selected_bank")
]
assert len(selected) == 1
assert int(selected.iloc[0]["bank_size"]) == int(bank_size)
assert str(selected.iloc[0]["bank_sha256"]) == bank_sha
manifest = json.loads(pathlib.Path(anchor_manifest).read_text())
assert manifest["outcome_blind"] is True and manifest["selection_rank_used"] is False
assert manifest["timing_counts_by_phase"] == {
    "development": {"rank_after_city": 10, "rank_before_city": 10},
    "confirmation": {"rank_after_city": 5, "rank_before_city": 5},
}
print(json.dumps({"model": model, "protocol_audit": "PASS"}, sort_keys=True))
PY

echo "WAIT_COMMIT_QUERY model=$MODEL path=$UPSTREAM_COMPLETE" | tee -a "$LOG"
while [[ ! -s "$UPSTREAM_COMPLETE" ]]; do
  sleep 30
done

run_phase() {
  local role=$1
  local phase=$2
  local mechanism=$3
  local trials="$OUTPUT_ROOT/$phase"
  local analysis="$OUTPUT_ROOT/analysis_$phase"
  echo "START model=$MODEL phase=$phase utc=$(date -u +%FT%TZ)" | tee -a "$LOG"
  CUDA_VISIBLE_DEVICES="$GPU_INDEX" "$PYTHON" "$RUNNER" \
    --mechanism-config "$mechanism" \
    --v5-config "$V5_CONFIG" \
    --model "$MODEL" \
    --cache-dir "$ROOT_DIR/work/hf_cache" \
    --device-map auto \
    --torch-dtype bfloat16 \
    --attention-backend sdpa \
    --generations "$GENERATIONS" \
    --seed-role "$role" \
    --anchor-registry "$ANCHORS" \
    --targeted-registry "$TARGETED_REGISTRY" \
    --bank-plan "$BANK_PLAN" \
    --source-layer "$SOURCE_LAYER" \
    --resume \
    --output "$trials" 2>&1 | tee -a "$LOG"
  "$PYTHON" "$ANALYZER" \
    --input "$trials" \
    --phase "$phase" \
    --random-seed 20260822 \
    --output "$analysis" 2>&1 | tee -a "$LOG"
  echo "SEALED model=$MODEL phase=$phase utc=$(date -u +%FT%TZ)" | tee -a "$LOG"
}

run_phase development discovery "$DEV_CONFIG"
run_phase confirmation confirmation "$CONFIRM_CONFIG"

"$PYTHON" - "$OUTPUT_ROOT/analysis_discovery/claim_gates.json" "$OUTPUT_ROOT/analysis_confirmation/claim_gates.json" "$COMPLETE" "$MODEL" "$BANK_SIZE" "$BANK_SHA" <<'PY'
import datetime as dt, json, os, pathlib, sys

discovery = json.loads(pathlib.Path(sys.argv[1]).read_text())
confirmation = json.loads(pathlib.Path(sys.argv[2]).read_text())
directional = bool(
    discovery["targeted_counter_write_directional_pass"]
    and confirmation["targeted_counter_write_directional_pass"]
)
strong = bool(
    discovery["targeted_counter_write_strong_gate_pass"]
    and confirmation["targeted_counter_write_strong_gate_pass"]
)
value = {
    "schema_version": "realistic_niah_v5_targeted_counter_write_complete_v1",
    "status": "PASS",
    "model_label": sys.argv[4],
    "completed_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
    "discovery_seed_count": 20,
    "confirmation_seed_count": 10,
    "outcome_blind": True,
    "selection_rank_used": False,
    "targeted_bank_size": int(sys.argv[5]),
    "targeted_bank_sha256": sys.argv[6],
    "teacher_forced_trace_tokens": True,
    "complete_directional_pass": directional,
    "complete_strong_gate_pass": strong,
    "discovery": discovery,
    "confirmation": confirmation,
}
path = pathlib.Path(sys.argv[3])
temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
temporary.replace(path)
print(json.dumps({"directional": directional, "strong": strong}, sort_keys=True))
PY
