#!/usr/bin/env bash
set -euo pipefail

MODEL=${1:?usage: launch_noindex_n10_after_n3.sh MODEL GPU N3_PID}
GPU=${2:?usage: launch_noindex_n10_after_n3.sh MODEL GPU N3_PID}
N3_PID=${3:?usage: launch_noindex_n10_after_n3.sh MODEL GPU N3_PID}
FS=${FS:-/lambda/nfs/CoT-Native-thinking-v5}
REPO=${REPO:-$FS/code/Realistic_CoT_NiaH_Count}
PY=$FS/venv/bin/python
ROOT=$REPO/work/natural_noindex_counter_n3_scan_20260826
N3=$ROOT/$MODEL
N10=$REPO/work/natural_noindex_counter_n10_scan_20260826/$MODEL
LOGS=$REPO/work/natural_noindex_counter_n10_scan_20260826/logs
mkdir -p "$LOGS"
LOG=$LOGS/${MODEL}.log
STATUS=$LOGS/${MODEL}.status.json
exec >>"$LOG" 2>&1

write_status() {
  local state=$1
  local detail=${2:-}
  printf '{"state":"%s","model":"%s","gpu":%s,"n3_pid":%s,"detail":"%s","utc":"%s"}\n' \
    "$state" "$MODEL" "$GPU" "$N3_PID" "$detail" "$(date -u +%FT%TZ)" >"$STATUS"
}

write_status waiting_for_n3
while kill -0 "$N3_PID" 2>/dev/null; do
  sleep 30
done

"$PY" -c 'import json,sys; x=json.load(open(sys.argv[1])); assert x["status"]=="FROZEN"; assert len(x["discovery_seeds"])==20; assert len(x["confirmation_seeds"])==10' "$N3/cohort_manifest_prefix_clean_v4.json"
write_status running_n10 "n3 cohort frozen"

case "$MODEL" in
  Qwen3-8B)
    SOURCES=(
      "$FS/runs/v5_native_thinking_representation/Qwen3-8B/generations.jsonl"
      "$FS/runs/v5_native_thinking_representation/supplement/Qwen3-8B/generations.jsonl"
      "$FS/runs/v5_native_thinking_representation/one_to_one_supplement/Qwen3-8B/batches/seed_1266_1271/generations.jsonl"
      "$FS/runs/v5_pure_trace_n10_supplement_20260824/native/Qwen3-8B/generations.jsonl"
    )
    ;;
  Gemma4-E4B)
    SOURCES=(
      "$FS/runs/v5_native_thinking_representation/Gemma4-E4B/generations.jsonl"
      "$FS/runs/v5_native_thinking_representation/marker_pair_supplement/Gemma4-E4B/generations.jsonl"
      "$FS/runs/v5_native_thinking_representation/one_to_one_supplement/Gemma4-E4B/batches/seed_1264_1293/generations.jsonl"
      "$FS/runs/v5_native_thinking_representation/one_to_one_supplement/Gemma4-E4B/batches/seed_1294_1323/generations.jsonl"
      "$FS/runs/v5_native_thinking_representation/one_to_one_supplement/Gemma4-E4B/batches/seed_1324_1353/generations.jsonl"
      "$FS/runs/v5_pure_trace_n10_supplement_20260824/native/Gemma4-E4B/generations.jsonl"
    )
    ;;
  *)
    write_status failed "unsupported model"
    exit 2
    ;;
esac

cd "$REPO"
env CUDA_VISIBLE_DEVICES="$GPU" "$PY" -m scripts.scan_realistic_niah_v5_frozen_prompt_noindex_n10 \
  --model "$MODEL" \
  --cache-dir "$FS/cache/huggingface" \
  --device-map auto \
  --torch-dtype bfloat16 \
  --attention-backend sdpa \
  --existing-generations "${SOURCES[@]}" \
  --max-seed 10000 \
  --output "$N10" \
  --resume
write_status complete "n10 cohort frozen"
