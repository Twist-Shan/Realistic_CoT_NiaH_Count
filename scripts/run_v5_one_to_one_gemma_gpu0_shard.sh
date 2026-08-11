#!/usr/bin/env bash
set -euo pipefail

FS=${FS:-/home/ubuntu/CoT-Native-thinking-v5}
REPO=${REPO:-$FS/code/Realistic_CoT_NiaH_Count}
RUN=${RUN:-$FS/runs/v5_native_thinking_representation}
MODEL=Gemma4-E4B
GPU=0
SEED_START=10000
SEED_END=10029
SUP=$RUN/one_to_one_supplement/$MODEL
BATCH=$SUP/batches/seed_z${SEED_START}_${SEED_END}
DATA=$BATCH/dataset
GEN=$BATCH/generations.jsonl
PARSED=$BATCH/parsed.jsonl
CAPTURE=$BATCH/capture
LOGS=$BATCH/logs
STATUS=$SUP/logs/gpu0_shard.status.json
PY=$FS/venv/bin/python

mkdir -p "$LOGS" "$(dirname "$STATUS")"
exec >> "$SUP/logs/gpu0_shard.supervisor.log" 2>&1

write_status() {
  local state=$1
  local detail=${2:-}
  printf '{"state":"%s","model":"%s","gpu":%s,"utc":"%s","detail":"%s"}\n' \
    "$state" "$MODEL" "$GPU" "$(date -u +%FT%TZ)" "$detail" > "$STATUS"
}

fail() {
  local rc=$?
  write_status failed "rc=$rc"
  exit "$rc"
}
trap fail ERR

cd "$REPO"
export HF_HOME="$FS/cache/huggingface"
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

discovery=()
confirmation=()
for seed in $(seq "$SEED_START" "$SEED_END"); do
  if (( (seed - 1266) % 3 == 2 )); then
    confirmation+=("$seed")
  else
    discovery+=("$seed")
  fi
done
discovery_csv=$(IFS=,; echo "${discovery[*]}")
confirmation_csv=$(IFS=,; echo "${confirmation[*]}")

write_status running "batch=$(basename "$BATCH") build"
if [[ ! -f "$DATA/candidate_manifest.json" ]]; then
  "$PY" scripts/build_realistic_niah_v5_one_to_one_candidates.py \
    --output-dir "$DATA" \
    --cache-dir "$FS/cache/huggingface" \
    --discovery-seeds "$discovery_csv" \
    --confirmation-seeds "$confirmation_csv" \
    > "$LOGS/build.log" 2>&1
fi
expected=$("$PY" -c 'import json,sys; print(json.load(open(sys.argv[1]))["candidate_rows"])' "$DATA/candidate_manifest.json")
test "$expected" -eq 30

if [[ ! -f "$GEN" ]]; then
  write_status running "batch=$(basename "$BATCH") generation"
  CUDA_VISIBLE_DEVICES="$GPU" "$PY" scripts/run_realistic_niah_v5.py generate \
    --config "$DATA/realistic_niah_v5_one_to_one_candidates.json" \
    --model "$MODEL" \
    --stimuli "$DATA/stimuli_native_thinking_n10.jsonl" \
    --output "$GEN" \
    --cache-dir "$FS/cache/huggingface" \
    --device-map auto \
    --torch-dtype bfloat16 \
    --attention-backend sdpa \
    > "$LOGS/generate.log" 2>&1
fi
test "$(wc -l < "$GEN")" -eq "$expected"

if [[ ! -f "$PARSED" ]]; then
  write_status running "batch=$(basename "$BATCH") parse"
  "$PY" scripts/run_realistic_niah_v5.py parse --input "$GEN" --output "$PARSED" > "$LOGS/parse.log" 2>&1
fi
test "$(wc -l < "$PARSED")" -eq "$expected"

if [[ ! -f "$BATCH/batch.complete.json" ]]; then
  write_status running "batch=$(basename "$BATCH") capture"
  CUDA_VISIBLE_DEVICES="$GPU" "$PY" scripts/run_realistic_niah_v5.py capture \
    --config "$DATA/realistic_niah_v5_one_to_one_candidates.json" \
    --model "$MODEL" \
    --generations "$GEN" \
    --output "$CAPTURE" \
    --cache-dir "$FS/cache/huggingface" \
    --device-map auto \
    --torch-dtype bfloat16 \
    --attention-backend sdpa \
    > "$LOGS/capture.log" 2>&1
  captured=$(wc -l < "$CAPTURE/capture_index.jsonl")
  excluded=0
  [[ ! -f "$CAPTURE/capture_exclusions.jsonl" ]] || excluded=$(wc -l < "$CAPTURE/capture_exclusions.jsonl")
  test "$((captured + excluded))" -eq "$expected"
  printf '{"schema_version":"realistic_niah_v5_one_to_one_batch_complete_v1","model":"%s","seed_start":%s,"seed_end":%s,"expected":%s,"captured":%s,"excluded":%s,"utc":"%s","ledger_write":false}\n' \
    "$MODEL" "$SEED_START" "$SEED_END" "$expected" "$captured" "$excluded" "$(date -u +%FT%TZ)" > "$BATCH/batch.complete.json"
fi

# Deliberately do not invoke the auditor here. The main Gemma supervisor is the
# sole writer of attempt_ledger.jsonl and will include this completed, disjoint
# batch on its next serial audit.
write_status ready_for_main_audit "batch=$(basename "$BATCH") ledger_write=false"
