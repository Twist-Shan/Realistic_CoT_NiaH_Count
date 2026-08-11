#!/usr/bin/env bash
set -euo pipefail

FS=${FS:-/home/ubuntu/CoT-Native-thinking-v5}
REPO=${REPO:-$FS/code/Realistic_CoT_NiaH_Count}
RUN=${RUN:-$FS/runs/v5_native_thinking_representation}
MODEL=${MODEL:?MODEL is required}
GPU=${GPU:?GPU is required}
START_SEED=${START_SEED:-1266}
TARGET_DISCOVERY=${TARGET_DISCOVERY:-20}
TARGET_CONFIRMATION=${TARGET_CONFIRMATION:-10}

case "$MODEL" in
  Qwen3-8B) BATCH_SIZE=${BATCH_SIZE:-6} ;;
  Gemma4-E4B) BATCH_SIZE=${BATCH_SIZE:-30} ;;
  *) echo "unsupported MODEL=$MODEL" >&2; exit 2 ;;
esac

PY=$FS/venv/bin/python
SUP=$RUN/one_to_one_supplement/$MODEL
BATCHES=$SUP/batches
LOGS=$SUP/logs
PRIMARY=$RUN/$MODEL/capture/capture_index.jsonl
EXISTING=$RUN/supplement/$MODEL/capture/capture_index.jsonl
AUDIT=$SUP/supplement_audit.json
mkdir -p "$BATCHES" "$LOGS"
exec >> "$LOGS/supervisor.log" 2>&1

write_status() {
  local state=$1
  local detail=${2:-}
  printf '{"state":"%s","model":"%s","gpu":%s,"utc":"%s","detail":"%s"}\n' \
    "$state" "$MODEL" "$GPU" "$(date -u +%FT%TZ)" "$detail" \
    > "$LOGS/supervisor.status.json"
}

fail() {
  local rc=$?
  write_status failed "rc=$rc"
  exit "$rc"
}
trap fail ERR

audit_now() {
  local args=(
    scripts/audit_realistic_niah_v5_one_to_one_supplement.py
    --model "$MODEL"
    --primary-capture-index "$PRIMARY"
    --supplement-root "$SUP"
    --target-discovery "$TARGET_DISCOVERY"
    --target-confirmation "$TARGET_CONFIRMATION"
  )
  if [[ -f "$EXISTING" ]]; then
    args+=(--existing-supplement-capture-index "$EXISTING")
  fi
  "$PY" "${args[@]}" > "$LOGS/audit.latest.log" 2>&1
}

cd "$REPO"
export HF_HOME="$FS/cache/huggingface"
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

write_status running initial_audit
audit_now

while true; do
  read -r total_d total_c is_complete < <(
    "$PY" -c 'import json,sys; a=json.load(open(sys.argv[1])); t=a["strict_total"]; print(t["discovery"],t["confirmation"],int(a["complete"]))' "$AUDIT"
  )
  echo "[one-to-one] model=$MODEL discovery=$total_d confirmation=$total_c complete=$is_complete"
  if [[ "$is_complete" == 1 ]]; then
    write_status complete "discovery=$total_d confirmation=$total_c"
    exit 0
  fi

  next_seed=$(
    "$PY" -c 'import json,sys,pathlib; p=pathlib.Path(sys.argv[1]); start=int(sys.argv[2]); rows=[json.loads(x) for x in p.open() if x.strip()] if p.exists() else []; print(max([int(r["seed"]) for r in rows],default=start-1)+1)' "$SUP/attempt_ledger.jsonl" "$START_SEED"
  )
  end_seed=$((next_seed + BATCH_SIZE - 1))
  discovery=()
  confirmation=()
  for seed in $(seq "$next_seed" "$end_seed"); do
    if [[ "$seed" == 1264 ]]; then
      discovery+=("$seed")
    elif [[ "$seed" == 1265 ]]; then
      confirmation+=("$seed")
    elif (( (seed - 1266) % 3 == 2 )); then
      confirmation+=("$seed")
    else
      discovery+=("$seed")
    fi
  done
  discovery_csv=$(IFS=,; echo "${discovery[*]}")
  confirmation_csv=$(IFS=,; echo "${confirmation[*]}")
  if [[ -z "$discovery_csv" || -z "$confirmation_csv" ]]; then
    echo "batch lacks one registered split: $next_seed-$end_seed" >&2
    exit 3
  fi

  batch=$BATCHES/seed_${next_seed}_${end_seed}
  data=$batch/dataset
  gen=$batch/generations.jsonl
  parsed=$batch/parsed.jsonl
  capture=$batch/capture
  mkdir -p "$batch/logs"
  write_status running "batch=$(basename "$batch") build"

  if [[ ! -f "$data/candidate_manifest.json" ]]; then
    "$PY" scripts/build_realistic_niah_v5_one_to_one_candidates.py \
      --output-dir "$data" \
      --cache-dir "$FS/cache/huggingface" \
      --discovery-seeds "$discovery_csv" \
      --confirmation-seeds "$confirmation_csv" \
      > "$batch/logs/build.log" 2>&1
  fi
  expected=$(
    "$PY" -c 'import json,sys; print(json.load(open(sys.argv[1]))["candidate_rows"])' "$data/candidate_manifest.json"
  )

  if [[ ! -f "$gen" ]]; then
    write_status running "batch=$(basename "$batch") generation"
    CUDA_VISIBLE_DEVICES="$GPU" "$PY" scripts/run_realistic_niah_v5.py generate \
      --config "$data/realistic_niah_v5_one_to_one_candidates.json" \
      --model "$MODEL" \
      --stimuli "$data/stimuli_native_thinking_n10.jsonl" \
      --output "$gen" \
      --cache-dir "$FS/cache/huggingface" \
      --device-map auto \
      --torch-dtype bfloat16 \
      --attention-backend sdpa \
      > "$batch/logs/generate.log" 2>&1
  fi
  test "$(wc -l < "$gen")" -eq "$expected"

  if [[ ! -f "$parsed" ]]; then
    write_status running "batch=$(basename "$batch") parse"
    "$PY" scripts/run_realistic_niah_v5.py parse \
      --input "$gen" \
      --output "$parsed" \
      > "$batch/logs/parse.log" 2>&1
  fi
  test "$(wc -l < "$parsed")" -eq "$expected"

  if [[ ! -f "$batch/batch.complete.json" ]]; then
    write_status running "batch=$(basename "$batch") capture"
    CUDA_VISIBLE_DEVICES="$GPU" "$PY" scripts/run_realistic_niah_v5.py capture \
      --config "$data/realistic_niah_v5_one_to_one_candidates.json" \
      --model "$MODEL" \
      --generations "$gen" \
      --output "$capture" \
      --cache-dir "$FS/cache/huggingface" \
      --device-map auto \
      --torch-dtype bfloat16 \
      --attention-backend sdpa \
      > "$batch/logs/capture.log" 2>&1
    captured=$(wc -l < "$capture/capture_index.jsonl")
    excluded=0
    if [[ -f "$capture/capture_exclusions.jsonl" ]]; then
      excluded=$(wc -l < "$capture/capture_exclusions.jsonl")
    fi
    test "$((captured + excluded))" -eq "$expected"
    printf '{"schema_version":"realistic_niah_v5_one_to_one_batch_complete_v1","model":"%s","seed_start":%s,"seed_end":%s,"expected":%s,"captured":%s,"excluded":%s,"utc":"%s"}\n' \
      "$MODEL" "$next_seed" "$end_seed" "$expected" "$captured" "$excluded" "$(date -u +%FT%TZ)" \
      > "$batch/batch.complete.json"
  fi

  write_status running "batch=$(basename "$batch") audit"
  audit_now
done
