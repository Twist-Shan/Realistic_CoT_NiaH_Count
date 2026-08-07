#!/usr/bin/env bash
set -Eeuo pipefail

: "${PROJECT_ROOT:?Set PROJECT_ROOT to the isolated noise code directory}"
: "${RUN_ROOT:?Set RUN_ROOT to the isolated noise run directory}"
: "${SOURCE_RUN:?Set SOURCE_RUN to the read-only v26 source run}"
: "${V4_PYTHON:?Set V4_PYTHON to the pinned environment Python}"

required=(
  "$RUN_ROOT/trace/extract_audit.json"
  "$RUN_ROOT/prompt/Qwen3-8B/capture/capture_manifest.json"
  "$RUN_ROOT/prompt/Gemma4-E4B/capture/capture_manifest.json"
)
while true; do
  missing=0
  for path in "${required[@]}"; do
    [[ -f "$path" ]] || missing=1
  done
  [[ "$missing" -eq 0 ]] && break
  sleep 60
done

cd "$PROJECT_ROOT"
export PYTHONPATH="$PROJECT_ROOT/src"
mkdir -p "$RUN_ROOT/analysis/prompt" "$RUN_ROOT/analysis/trace_factors" \
  "$RUN_ROOT/analysis/prompt_factors" "$RUN_ROOT/analysis/comparison" "$RUN_ROOT/logs"
mkdir -p "$RUN_ROOT/analysis/trace_factors_correct" \
  "$RUN_ROOT/analysis/prompt_factors_correct" "$RUN_ROOT/analysis/decomposition"

for model in Qwen3-8B Gemma4-E4B; do
  "$V4_PYTHON" scripts/analyze_realistic_niah_v4_4_native_prompt_noise.py \
    --capture "$RUN_ROOT/prompt/$model/capture" \
    --requests "$SOURCE_RUN/baseline/$model/requests.jsonl" \
    --output "$RUN_ROOT/analysis/prompt/$model" --rank 3 \
    > "$RUN_ROOT/logs/prompt_noise_${model}.log" 2>&1
done

"$V4_PYTHON" scripts/analyze_realistic_niah_v4_4_native_noise_factors.py \
  --inputs "$RUN_ROOT/trace/Qwen3-8B/trace_noise_rows.csv.gz" \
    "$RUN_ROOT/trace/Gemma4-E4B/trace_noise_rows.csv.gz" \
  --output "$RUN_ROOT/analysis/trace_factors" --split confirmation \
  --sample-fraction 0.20 --max-rows 150000 --folds 5 --n-jobs 8 \
  --targets noise_total_rms count_axis_deviation_abs \
  --algorithms elastic_net hist_gradient_boosting \
  > "$RUN_ROOT/logs/trace_noise_factors.log" 2>&1

"$V4_PYTHON" scripts/analyze_realistic_niah_v4_4_native_noise_factors.py \
  --inputs "$RUN_ROOT/trace/Qwen3-8B/trace_noise_rows.csv.gz" \
    "$RUN_ROOT/trace/Gemma4-E4B/trace_noise_rows.csv.gz" \
  --output "$RUN_ROOT/analysis/trace_factors_correct" --split confirmation \
  --correct-only --sample-fraction 0.25 --max-rows 150000 --folds 5 --n-jobs 8 \
  --targets noise_total_rms count_axis_deviation_abs \
  --algorithms elastic_net hist_gradient_boosting \
  > "$RUN_ROOT/logs/trace_noise_factors_correct.log" 2>&1

"$V4_PYTHON" scripts/analyze_realistic_niah_v4_4_native_prompt_noise_factors.py \
  --inputs "$RUN_ROOT/analysis/prompt/Qwen3-8B/prompt_noise_rows.csv.gz" \
    "$RUN_ROOT/analysis/prompt/Gemma4-E4B/prompt_noise_rows.csv.gz" \
  --output "$RUN_ROOT/analysis/prompt_factors" --split confirmation \
  --folds 5 --n-jobs 8 \
  > "$RUN_ROOT/logs/prompt_noise_factors.log" 2>&1

"$V4_PYTHON" scripts/analyze_realistic_niah_v4_4_native_prompt_noise_factors.py \
  --inputs "$RUN_ROOT/analysis/prompt/Qwen3-8B/prompt_noise_rows.csv.gz" \
    "$RUN_ROOT/analysis/prompt/Gemma4-E4B/prompt_noise_rows.csv.gz" \
  --output "$RUN_ROOT/analysis/prompt_factors_correct" --split confirmation \
  --correct-only --folds 5 --n-jobs 8 \
  > "$RUN_ROOT/logs/prompt_noise_factors_correct.log" 2>&1

"$V4_PYTHON" scripts/compare_realistic_niah_v4_4_prompt_trace_noise.py \
  --source-run "$SOURCE_RUN" --trace-root "$RUN_ROOT/trace" \
  --prompt-root "$RUN_ROOT/analysis/prompt" \
  --output "$RUN_ROOT/analysis/comparison" \
  > "$RUN_ROOT/logs/prompt_trace_comparison.log" 2>&1

"$V4_PYTHON" scripts/summarize_realistic_niah_v4_4_correct_noise_decomposition.py \
  --trace-inputs "$RUN_ROOT/trace/Qwen3-8B/trace_noise_rows.csv.gz" \
    "$RUN_ROOT/trace/Gemma4-E4B/trace_noise_rows.csv.gz" \
  --prompt-inputs "$RUN_ROOT/analysis/prompt/Qwen3-8B/prompt_noise_rows.csv.gz" \
    "$RUN_ROOT/analysis/prompt/Gemma4-E4B/prompt_noise_rows.csv.gz" \
  --output "$RUN_ROOT/analysis/decomposition" \
  > "$RUN_ROOT/logs/noise_decomposition.log" 2>&1

date --iso-8601=seconds > "$RUN_ROOT/noise_postprocess.complete"
