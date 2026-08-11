#!/usr/bin/env bash
set -euo pipefail

MODEL=${1:?model label required}
GPU=${2:?gpu index required}
DISCOVERY_SEEDS=${DISCOVERY_SEEDS:?comma-separated discovery seeds required}
CONFIRMATION_SEEDS=${CONFIRMATION_SEEDS:?comma-separated confirmation seeds required}
COUNTS=${COUNTS:-3,4,9}
FS=${FS:-/home/ubuntu/CoT-Native-thinking-v5}
REPO=${REPO:-$FS/code/Realistic_CoT_NiaH_Count}
RUN=${RUN:-$FS/runs/v5_native_thinking_representation}
PY=$FS/venv/bin/python
CACHE=$FS/cache/huggingface
ROOT=$RUN/marker_pair_supplement/$MODEL
DATA=$ROOT/dataset
GEN=$ROOT/generations.jsonl
PARSED=$ROOT/parsed.jsonl
PLAN=$ROOT/combined_marker_plan
LOG=$ROOT/logs
PRIMARY=$RUN/$MODEL/generations.jsonl

mkdir -p "$LOG"
cd "$REPO"
printf 'running\n' > "$LOG/supervisor.status"
printf '%s\n' "$$" > "$LOG/supervisor.pid"
trap 'code=$?; if [[ $code -ne 0 ]]; then printf "failed:%s\n" "$code" > "$LOG/supervisor.status"; fi' EXIT

if [[ ! -f "$DATA/candidate_manifest.json" ]]; then
  "$PY" scripts/build_v5_marker_pair_supplement_candidates.py \
    --output-dir "$DATA" --cache-dir "$CACHE" \
    --discovery-seeds "$DISCOVERY_SEEDS" \
    --confirmation-seeds "$CONFIRMATION_SEEDS" --counts "$COUNTS" \
    > "$LOG/build.log" 2>&1
fi
EXPECTED=$("$PY" -c 'import json,sys; print(json.load(open(sys.argv[1]))["candidate_rows"])' "$DATA/candidate_manifest.json")

if [[ ! -f "$GEN" ]]; then
  CUDA_VISIBLE_DEVICES="$GPU" "$PY" scripts/run_realistic_niah_v5.py generate \
    --config "$DATA/realistic_niah_v5_marker_pair_supplement.json" \
    --model "$MODEL" --stimuli "$DATA/stimuli_native_thinking_marker_pairs.jsonl" \
    --output "$GEN" --cache-dir "$CACHE" --device-map auto \
    --torch-dtype bfloat16 --attention-backend sdpa \
    > "$LOG/generate.log" 2>&1
fi
test "$(wc -l < "$GEN")" -eq "$EXPECTED"

if [[ ! -f "$PARSED" ]]; then
  "$PY" scripts/run_realistic_niah_v5.py parse --input "$GEN" --output "$PARSED" \
    > "$LOG/parse.log" 2>&1
fi
test "$(wc -l < "$PARSED")" -eq "$EXPECTED"

GEN_INPUTS=("$PRIMARY" "$GEN")
while IFS= read -r source; do
  GEN_INPUTS+=("$source")
done < <(find "$RUN/one_to_one_supplement/$MODEL/batches" -type f -name generations.jsonl | sort)

"$PY" scripts/build_v5_marker_needle_patch_plan.py \
  --generations "${GEN_INPUTS[@]}" --model "$MODEL" --output-dir "$PLAN" \
  > "$LOG/combined_plan.log" 2>&1
"$PY" scripts/audit_v5_marker_pair_coverage.py \
  --pairs "$PLAN/${MODEL}__marker_adjacent_pairs.jsonl" \
  --output "$ROOT/coverage_audit.json" > "$LOG/coverage_audit.log" 2>&1

printf 'complete\n' > "$LOG/supervisor.status"
printf '[%s] complete model=%s expected=%s\n' "$(date -Is)" "$MODEL" "$EXPECTED" \
  >> "$LOG/supervisor.log"
