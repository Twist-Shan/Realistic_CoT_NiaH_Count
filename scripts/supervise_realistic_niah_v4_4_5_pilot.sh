#!/usr/bin/env bash
set -Eeuo pipefail

repo_root=${1:?repo root required}
run_root=${2:?run root required}
qwen_pid=${3:?Qwen PID required}
python_bin=${4:?Python executable required}
hf_cache=${5:?HF cache required}

while kill -0 "$qwen_pid" 2>/dev/null; do
  sleep 30
done

qwen_complete="$run_root/pilot/Qwen3-8B/complete.json"
gemma_complete="$run_root/pilot/Gemma4-E4B/complete.json"
if [[ ! -s "$qwen_complete" ]]; then
  echo "Qwen exited without complete.json; refusing to start Gemma" >&2
  exit 1
fi
if [[ -s "$gemma_complete" ]]; then
  echo "Gemma already complete"
  exit 0
fi

cd "$repo_root"
env \
  HF_HOME="$hf_cache" \
  TOKENIZERS_PARALLELISM=false \
  PYTHONPATH="$repo_root/src" \
  "$python_bin" scripts/run_realistic_niah_v4_4_5_span_restoration.py \
    --model Gemma4-E4B \
    --stimuli "$run_root/dataset/stimuli_v4_4_causal_v2.jsonl" \
    --output-dir "$run_root/pilot" \
    --cache-dir "$hf_cache" \
    > "$run_root/logs/gemma_pilot.log" 2>&1
