#!/usr/bin/env bash
set -euo pipefail

repo="/home/ubuntu/CoT-Native-thinking-v5/code/Realistic_CoT_NiaH_Count_native_v5_20260818"
python_bin="/lambda/nfs/CoT-Native-thinking-v5/venv/bin/python"
run_root="/home/ubuntu/CoT-Native-thinking-v5/runs/v5_gemma_prefix_record_control_20260824"
cache_root="/home/ubuntu/CoT-Native-thinking-v5/hf_cache"

cd "$repo"

"$python_bin" scripts/run_v5_gemma_prefix_record_control.py \
  --config "$run_root/candidates/realistic_niah_v5_one_to_one_candidates.json" \
  --stimuli "$run_root/candidates/stimuli_native_thinking_n10.jsonl" \
  --stimuli "$run_root/candidates2/stimuli_native_thinking_n10.jsonl" \
  --stimuli "$run_root/candidates3/stimuli_native_thinking_n10.jsonl" \
  --output "$run_root/native/Gemma4-E4B/generations.jsonl" \
  --accepted-output "$run_root/accepted.jsonl" \
  --manifest "$run_root/manifest.json" \
  --target-accepted 30 \
  --cache-dir "$cache_root" \
  --device-map cuda:0 \
  --torch-dtype bfloat16

"$python_bin" scripts/select_v5_gemma_prefix_record_control.py \
  --accepted "$run_root/accepted.jsonl" \
  --output "$run_root/selection_20_10.csv" \
  --generations "$run_root/native/Gemma4-E4B/generations.jsonl" \
  --selected-generations-output "$run_root/selected_generations_20_10.jsonl"

"$python_bin" scripts/capture_v5_pure_trace_n10_paired_supplement.py \
  --model Gemma4-E4B \
  --selection "$run_root/selection_20_10.csv" \
  --generations "$run_root/native/Gemma4-E4B/generations.jsonl" \
  --stimuli "$run_root/candidates/stimuli_native_thinking_n10.jsonl" \
  --stimuli "$run_root/candidates2/stimuli_native_thinking_n10.jsonl" \
  --stimuli "$run_root/candidates3/stimuli_native_thinking_n10.jsonl" \
  --output-root "$run_root/capture" \
  --cache-dir "$cache_root" \
  --device-map cuda:0 \
  --torch-dtype bfloat16 \
  --native-site-kinds item_end

"$python_bin" -c "from pathlib import Path; Path('$run_root/PIPELINE_DONE').write_text('generation, split, and capture complete\n', encoding='utf-8')"
