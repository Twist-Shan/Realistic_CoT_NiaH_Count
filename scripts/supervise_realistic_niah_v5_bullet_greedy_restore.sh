#!/usr/bin/env bash
set -euo pipefail

repo=/home/ubuntu/bullet_counter_exp_20260824
nfs=/home/ubuntu/CoT-Native-thinking-v5
python_bin="$repo/.venv/bin/python"
cohort_root="$nfs/runs/v5_marker_scrubbed_list_counterfactual_restore_20260824"
output_root="$nfs/runs/v5_marker_scrubbed_list_greedy_restore_20260824/corrected_same_cohort"

export PYTHONDONTWRITEBYTECODE=1
cd "$repo"

run_model() {
    local model=$1
    shift
    local discovery_layers=("$@")
    local model_output="$output_root/$model"
    mkdir -p "$model_output"

    "$python_bin" scripts/run_realistic_niah_v5_bullet_greedy_restore.py \
        --model "$model" \
        --cache-dir "$nfs/hf_cache" \
        --device-map auto \
        --torch-dtype bfloat16 \
        --attention-backend sdpa \
        --generations "$cohort_root/$model/cohort/eligible_generations.jsonl" \
        --cohort-manifest "$cohort_root/$model/cohort/frozen_cohort_manifest.json" \
        --phase discovery \
        --source-layers "${discovery_layers[@]}" \
        --max-new-tokens 2 \
        --output "$model_output/discovery"

    "$python_bin" scripts/analyze_realistic_niah_v5_bullet_greedy_restore.py \
        --input "$model_output/discovery" \
        --phase discovery \
        --output "$model_output/discovery/analysis"

    mapfile -t frozen_layers < <(
        "$python_bin" -c \
            "import json; p=json.load(open('$model_output/discovery/analysis/frozen_layers.json')); print(*p['source_layers'], sep='\n')"
    )
    if [[ ${#frozen_layers[@]} -ne 3 ]]; then
        echo "Expected exactly three greedy-frozen layers for $model" >&2
        exit 1
    fi

    "$python_bin" scripts/run_realistic_niah_v5_bullet_greedy_restore.py \
        --model "$model" \
        --cache-dir "$nfs/hf_cache" \
        --device-map auto \
        --torch-dtype bfloat16 \
        --attention-backend sdpa \
        --generations "$cohort_root/$model/cohort/eligible_generations.jsonl" \
        --cohort-manifest "$cohort_root/$model/cohort/frozen_cohort_manifest.json" \
        --phase confirmation \
        --source-layers "${frozen_layers[@]}" \
        --max-new-tokens 2 \
        --output "$model_output/confirmation"

    "$python_bin" scripts/analyze_realistic_niah_v5_bullet_greedy_restore.py \
        --input "$model_output/confirmation" \
        --phase confirmation \
        --frozen-layers "$model_output/discovery/analysis/frozen_layers.json" \
        --output "$model_output/confirmation/analysis"
}

run_model Qwen3-8B 0 4 8 12 16 20 24 28 32
run_model Gemma4-E4B 0 4 8 12 16 20 24 28 32 36 40
printf 'PASS\n' > "$output_root/DONE"
