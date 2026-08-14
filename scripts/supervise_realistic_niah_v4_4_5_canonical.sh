#!/usr/bin/env bash
set -euo pipefail

repo="/home/ubuntu/Realistic_CoT_NiaH_Count"
root="/home/ubuntu/runs/nonthinking_v445_20260813"
stimuli="$root/dataset/canonical_run_20260731_v4_numeric_presentation_v3_stimuli.jsonl"
output="$root/canonical_span_restoration"
analysis="$root/canonical_span_restoration_analysis"
cache="/home/ubuntu/hf-cache"
logs="$root/logs"
seeds="1234,1235,1236,1237,1238,1239,1240,1241,1242,1243,1244,1245,1246,1247,1248,1249,1250,1251,1252,1253,1254,1255,1256,1257,1258,1259,1260,1261,1262,1263"
counts="1,2,3,4,5,6,7,8,9,10"
expected_stimuli_sha="da4dd86142eb8a07f9a7e53497efd3375184c8e68367d4db994370fcb331f090"
qwen_expected_rows=33300
gemma_expected_rows=38700

mkdir -p "$output" "$analysis" "$logs"
cd "$repo"

actual_stimuli_sha="$(sha256sum "$stimuli" | cut -d' ' -f1)"
if [[ "$actual_stimuli_sha" != "$expected_stimuli_sha" ]]; then
  echo "Canonical stimulus hash audit failed: $actual_stimuli_sha != $expected_stimuli_sha" >&2
  exit 1
fi

PYTHONPATH=src .venv/bin/python \
  scripts/run_realistic_niah_v4_4_5_span_restoration.py \
  --model Qwen3-8B \
  --stimuli "$stimuli" \
  --stimuli-config configs/realistic_niah_v4.json \
  --experiment-config configs/realistic_niah_v4_4_5_span_restoration_canonical.json \
  --output-dir "$output" \
  --cache-dir "$cache" \
  --seeds "$seeds" \
  --counts "$counts" \
  --layers "0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35" \
  --patch-kinds "needle_endpoint,needle_full,ordinary_full" \
  --skip-cache-equivalence-audit \
  >"$logs/qwen_canonical_span_restoration.log" 2>&1

qwen_rows="$(wc -l < "$output/Qwen3-8B/detail.jsonl")"
if [[ "$qwen_rows" -ne "$qwen_expected_rows" ]]; then
  echo "Qwen canonical row audit failed: $qwen_rows != $qwen_expected_rows" >&2
  exit 1
fi

PYTHONPATH=src .venv/bin/python \
  scripts/run_realistic_niah_v4_4_5_span_restoration.py \
  --model Gemma4-E4B \
  --stimuli "$stimuli" \
  --stimuli-config configs/realistic_niah_v4.json \
  --experiment-config configs/realistic_niah_v4_4_5_span_restoration_canonical.json \
  --output-dir "$output" \
  --cache-dir "$cache" \
  --seeds "$seeds" \
  --counts "$counts" \
  --layers "0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41" \
  --patch-kinds "needle_endpoint,needle_full,ordinary_full" \
  --skip-cache-equivalence-audit \
  >"$logs/gemma_canonical_span_restoration.log" 2>&1

gemma_rows="$(wc -l < "$output/Gemma4-E4B/detail.jsonl")"
if [[ "$gemma_rows" -ne "$gemma_expected_rows" ]]; then
  echo "Gemma canonical row audit failed: $gemma_rows != $gemma_expected_rows" >&2
  exit 1
fi

PYTHONPATH=src .venv/bin/python \
  scripts/analyze_realistic_niah_v4_4_5_span_restoration.py \
  --run-root "$output" \
  --output-dir "$analysis" \
  --experiment-config configs/realistic_niah_v4_4_5_span_restoration_canonical.json \
  --models Qwen3-8B Gemma4-E4B \
  >"$logs/canonical_span_restoration_analysis.log" 2>&1

cat >"$root/canonical_span_restoration_supervisor_complete.json" <<EOF
{
  "status": "complete",
  "stimuli_sha256": "$actual_stimuli_sha",
  "seeds": 30,
  "counts": 10,
  "qwen_rows": $qwen_rows,
  "gemma_rows": $gemma_rows,
  "analysis": "$analysis"
}
EOF
