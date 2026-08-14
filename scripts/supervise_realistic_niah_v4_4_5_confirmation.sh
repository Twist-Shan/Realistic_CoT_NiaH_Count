#!/usr/bin/env bash
set -euo pipefail

repo="/home/ubuntu/Realistic_CoT_NiaH_Count"
root="/home/ubuntu/runs/nonthinking_v445_20260813"
stimuli="$root/dataset/stimuli_v4_4_causal_v2.jsonl"
output="$root/confirmation"
analysis="$root/confirmation_analysis"
cache="/home/ubuntu/hf-cache"
logs="$root/logs"
seeds="2010,2011,2012,2013,2014,2015,2016,2017,2018,2019"
counts="3,6,9"

mkdir -p "$output" "$analysis" "$logs"
cd "$repo"

PYTHONPATH=src .venv/bin/python \
  scripts/run_realistic_niah_v4_4_5_span_restoration.py \
  --model Qwen3-8B \
  --stimuli "$stimuli" \
  --output-dir "$output" \
  --cache-dir "$cache" \
  --seeds "$seeds" \
  --counts "$counts" \
  --layers "0,8,16,20,24,28" \
  --patch-kinds "needle_endpoint,needle_full,ordinary_full" \
  >"$logs/qwen_confirmation.log" 2>&1

qwen_rows="$(wc -l < "$output/Qwen3-8B/detail.jsonl")"
if [[ "$qwen_rows" -ne 630 ]]; then
  echo "Qwen confirmation row audit failed: $qwen_rows != 630" >&2
  exit 1
fi

PYTHONPATH=src .venv/bin/python \
  scripts/run_realistic_niah_v4_4_5_span_restoration.py \
  --model Gemma4-E4B \
  --stimuli "$stimuli" \
  --output-dir "$output" \
  --cache-dir "$cache" \
  --seeds "$seeds" \
  --counts "$counts" \
  --layers "0,8,12,16,20,24" \
  --patch-kinds "needle_endpoint,needle_full,ordinary_full" \
  >"$logs/gemma_confirmation.log" 2>&1

gemma_rows="$(wc -l < "$output/Gemma4-E4B/detail.jsonl")"
if [[ "$gemma_rows" -ne 630 ]]; then
  echo "Gemma confirmation row audit failed: $gemma_rows != 630" >&2
  exit 1
fi

PYTHONPATH=src .venv/bin/python \
  scripts/analyze_realistic_niah_v4_4_5_span_restoration.py \
  --run-root "$output" \
  --output-dir "$analysis" \
  --models Qwen3-8B Gemma4-E4B \
  >"$logs/confirmation_analysis.log" 2>&1

cat >"$root/confirmation_supervisor_complete.json" <<EOF
{
  "status": "complete",
  "qwen_rows": $qwen_rows,
  "gemma_rows": $gemma_rows,
  "analysis": "$analysis"
}
EOF
