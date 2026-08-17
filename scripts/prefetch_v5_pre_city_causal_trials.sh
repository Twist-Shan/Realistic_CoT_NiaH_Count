#!/usr/bin/env bash
set -euo pipefail

if [[ "$#" -lt 4 ]]; then
  echo "usage: $0 <model> <gpu-index> <query-variant> <K> [K ...]" >&2
  exit 2
fi

FS=/home/ubuntu/CoT-Native-thinking-v5
REPO=$FS/code/Realistic_CoT_NiaH_Count
RUN=$FS/runs/v5_native_thinking_representation
MODEL=$1
GPU=$2
VARIANT=$3
shift 3
BANK_SIZES=("$@")
MODEL_DIR=$RUN/$MODEL
OUT_DIR=$MODEL_DIR/causal/pre_city_token/head_tests
PLAN=$MODEL_DIR/causal/pre_city_token/plan/causal_plan.csv
MANIFEST=$OUT_DIR/chunk_manifest.tsv
TRIAL_DIR=$OUT_DIR/trials
LOG_DIR=$MODEL_DIR/logs
STATUS=$LOG_DIR/pre_city_causal_prefetch_gpu${GPU}.status

mkdir -p "$TRIAL_DIR" "$LOG_DIR"
exec >> "$LOG_DIR/pre_city_causal_prefetch_gpu${GPU}.log" 2>&1

fail() {
  rc=$?
  printf 'failed rc=%s utc=%s\n' "$rc" "$(date -u +%FT%TZ)" > "$STATUS"
  exit "$rc"
}
trap fail ERR

cd "$REPO"
export HF_HOME="$FS/cache/huggingface"
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

test -s "$PLAN"
test -s "$MANIFEST"
test -s "$MODEL_DIR/generations.jsonl"

for bank_size in "${BANK_SIZES[@]}"; do
  line=$(awk -F $'\t' -v variant="$VARIANT" -v k="$bank_size" \
    '$1 == variant && $2 == k { print; exit }' "$MANIFEST")
  if [[ -z "$line" ]]; then
    echo "manifest has no chunk variant=$VARIANT K=$bank_size" >&2
    exit 1
  fi
  IFS=$'\t' read -r _variant _bank_size indices _has_random <<< "$line"
  trial=$TRIAL_DIR/${VARIANT}_K${bank_size}.jsonl
  if [[ -s "$trial" ]]; then
    echo "[$MODEL pre-city prefetch] reuse variant=$VARIANT K=$bank_size"
    continue
  fi
  temporary=$TRIAL_DIR/.${VARIANT}_K${bank_size}.gpu${GPU}.tmp.jsonl
  rm -f "$temporary"
  printf 'running utc=%s variant=%s K=%s output=%s\n' \
    "$(date -u +%FT%TZ)" "$VARIANT" "$bank_size" "$temporary" > "$STATUS"
  # shellcheck disable=SC2086
  CUDA_VISIBLE_DEVICES="$GPU" "$FS/venv/bin/python" \
    scripts/run_realistic_niah_v5.py causal-pre-city-heads \
    --model "$MODEL" \
    --generations "$MODEL_DIR/generations.jsonl" \
    --plan "$PLAN" \
    --plan-rows $indices \
    --query-variant "$VARIANT" \
    --output "$temporary" \
    --cohort one_to_one \
    --cache-dir "$FS/cache/huggingface" \
    --device-map auto \
    --torch-dtype bfloat16 \
    --attention-backend sdpa \
    > "$LOG_DIR/pre_city_causal_prefetch_${VARIANT}_K${bank_size}_gpu${GPU}.log" 2>&1
  test -s "$temporary"
  if [[ -s "$trial" ]]; then
    echo "final trial appeared during prefetch; preserving supervisor output" >&2
    rm -f "$temporary"
  else
    mv "$temporary" "$trial"
  fi
  echo "[$MODEL pre-city prefetch] complete variant=$VARIANT K=$bank_size"
done

printf 'complete utc=%s variant=%s banks=%s\n' \
  "$(date -u +%FT%TZ)" "$VARIANT" "${BANK_SIZES[*]}" > "$STATUS"
