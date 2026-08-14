#!/usr/bin/env bash
set -euo pipefail

repo="/home/ubuntu/Realistic_CoT_NiaH_Count"
root="/home/ubuntu/runs/nonthinking_v445_20260813"
stimuli="$root/dataset/canonical_run_20260731_v4_numeric_presentation_v3_stimuli.jsonl"
output="$root/canonical_span_restoration"
answer_analysis="$root/canonical_answer_geometry_analysis"
retrieval_analysis="$root/canonical_retrieval_geometry_analysis"
logs="$root/logs"
expected_stimuli_sha="da4dd86142eb8a07f9a7e53497efd3375184c8e68367d4db994370fcb331f090"

mkdir -p "$output" "$answer_analysis" "$retrieval_analysis" "$logs"
cd "$repo"

actual_stimuli_sha="$(sha256sum "$stimuli" | cut -d' ' -f1)"
if [[ "$actual_stimuli_sha" != "$expected_stimuli_sha" ]]; then
  echo "Canonical stimulus hash audit failed: $actual_stimuli_sha != $expected_stimuli_sha" >&2
  exit 1
fi

test -f "$root/canonical_qwen_dense_complete.json"
test -f "$root/canonical_gemma_dense_complete.json"
qwen_rows="$(wc -l < "$output/Qwen3-8B/detail.jsonl")"
gemma_rows="$(wc -l < "$output/Gemma4-E4B/detail.jsonl")"
if [[ "$qwen_rows" -ne 33300 || "$gemma_rows" -ne 38700 ]]; then
  echo "Dense-run geometry input audit failed: Qwen=$qwen_rows Gemma=$gemma_rows" >&2
  exit 1
fi

PYTHONPATH=src .venv/bin/python \
  scripts/analyze_realistic_niah_v4_4_5_answer_geometry.py \
  --run-root "$output" \
  --output-dir "$answer_analysis" \
  --experiment-config configs/realistic_niah_v4_4_5_span_restoration_canonical.json \
  --models Qwen3-8B Gemma4-E4B \
  >"$logs/answer_geometry_analysis.log" 2>&1

PYTHONPATH=src .venv/bin/python \
  scripts/analyze_realistic_niah_v4_4_5_retrieval_geometry.py \
  --run-root "$output" \
  --output-dir "$retrieval_analysis" \
  --experiment-config configs/realistic_niah_v4_4_5_span_restoration_canonical.json \
  --models Qwen3-8B Gemma4-E4B \
  --bootstrap-draws 200 \
  >"$logs/retrieval_geometry_analysis.log" 2>&1

cat >"$root/canonical_geometry_supervisor_complete.json" <<EOF
{
  "status": "complete",
  "stimuli_sha256": "$actual_stimuli_sha",
  "source": "canonical_span_restoration clean baseline rows (no duplicate GPU run)",
  "qwen_detail_rows": $qwen_rows,
  "gemma_detail_rows": $gemma_rows,
  "clean_state_rows_per_model": 300,
  "answer_analysis": "$answer_analysis",
  "retrieval_analysis": "$retrieval_analysis"
}
EOF
