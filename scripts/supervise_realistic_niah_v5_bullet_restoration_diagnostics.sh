#!/usr/bin/env bash
set -euo pipefail

repo=/home/ubuntu/bullet_counter_exp_20260824
nfs=/home/ubuntu/CoT-Native-thinking-v5
python_bin="$repo/.venv/bin/python"
run_root="$nfs/runs/v5_marker_scrubbed_list_counterfactual_restore_20260824"
output_root="$nfs/runs/v5_marker_scrubbed_list_restoration_diagnostics_20260824/panel_3seed_k2369"

export PYTHONDONTWRITEBYTECODE=1
cd "$repo"

run_model() {
    local model=$1
    local layer=$2
    shift 2
    local seeds=("$@")
    local output="$output_root/$model"
    mkdir -p "$output"
    "$python_bin" scripts/run_realistic_niah_v5_bullet_restoration_diagnostics.py \
        --model "$model" \
        --cache-dir "$nfs/hf_cache" \
        --device-map auto \
        --torch-dtype bfloat16 \
        --attention-backend sdpa \
        --generations "$run_root/$model/cohort/eligible_generations.jsonl" \
        --cohort-manifest "$run_root/$model/cohort/frozen_cohort_manifest.json" \
        --seeds "${seeds[@]}" \
        --source-layer "$layer" \
        --target-occurrences 2 3 6 9 \
        --output "$output"
    "$python_bin" scripts/analyze_realistic_niah_v5_bullet_restoration_diagnostics.py \
        --input "$output" \
        --output "$output/analysis"
}

run_model Qwen3-8B 16 1288 1289 1290
run_model Gemma4-E4B 20 1282 1283 1284
printf 'PASS\n' > "$output_root/DONE"
