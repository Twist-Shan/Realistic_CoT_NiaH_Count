#!/usr/bin/env bash
set -euo pipefail

repo="/home/ubuntu/Realistic_CoT_NiaH_Count"
root="/home/ubuntu/runs/nonthinking_v445_20260813"
stimuli="$root/dataset/canonical_run_20260731_v4_numeric_presentation_v3_stimuli.jsonl"
basis="$root/canonical_retrieval_geometry_analysis/retrieval_bases.pt"
restoration="$root/canonical_span_restoration"
run_root="$root/canonical_retrieval_subspace"
analysis_root="$root/canonical_retrieval_subspace_analysis"
cache="/home/ubuntu/hf-cache"
logs="$root/logs"
seeds="1254,1255,1256,1257,1258,1259,1260,1261,1262,1263"
counts="1,2,3,4,5,6,7,8,9,10"
expected_stimuli_sha="da4dd86142eb8a07f9a7e53497efd3375184c8e68367d4db994370fcb331f090"

mkdir -p "$run_root" "$analysis_root" "$logs"
cd "$repo"
test -f "$basis"
actual_stimuli_sha="$(sha256sum "$stimuli" | cut -d' ' -f1)"
if [[ "$actual_stimuli_sha" != "$expected_stimuli_sha" ]]; then
  echo "Canonical stimulus hash audit failed: $actual_stimuli_sha != $expected_stimuli_sha" >&2
  exit 1
fi

run_one() {
  local model="$1"
  local retrieval_layer="$2"
  local label="$3"
  local output="$run_root/$label"
  local analysis="$analysis_root/$label"

  PYTHONPATH=src .venv/bin/python \
    scripts/run_realistic_niah_v4_4_5_retrieval_subspace.py \
    --model "$model" \
    --stimuli "$stimuli" \
    --stimuli-config configs/realistic_niah_v4.json \
    --experiment-config configs/realistic_niah_v4_4_5_span_restoration_canonical.json \
    --basis-file "$basis" \
    --retrieval-layer "$retrieval_layer" \
    --source-patch-layer 8 \
    --seeds "$seeds" \
    --counts "$counts" \
    --output-dir "$output" \
    --cache-dir "$cache" \
    >"$logs/${label}_retrieval_subspace.log" 2>&1

  local rows
  rows="$(wc -l < "$output/$model/detail.jsonl")"
  if [[ "$rows" -ne 400 ]]; then
    echo "$label retrieval-subspace row audit failed: $rows != 400" >&2
    exit 1
  fi

  PYTHONPATH=src .venv/bin/python \
    scripts/analyze_realistic_niah_v4_4_5_retrieval_subspace.py \
    --run-root "$output" \
    --restoration-root "$restoration" \
    --output-dir "$analysis" \
    --models "$model" \
    >"$logs/${label}_retrieval_subspace_analysis.log" 2>&1
}

# Frozen before looking at V4.4.5 confirmation geometry: the two layers with the
# largest frozen broad-head memberships in each pre-existing ranked bank.
run_one Qwen3-8B 21 Qwen_L21
run_one Qwen3-8B 23 Qwen_L23
run_one Qwen3-8B 24 Qwen_L24
run_one Qwen3-8B 26 Qwen_L26
run_one Qwen3-8B 27 Qwen_L27
run_one Gemma4-E4B 29 Gemma_L29
run_one Gemma4-E4B 35 Gemma_L35

cat >"$root/canonical_retrieval_subspace_supervisor_complete.json" <<EOF
{
  "status": "complete",
  "stimuli_sha256": "$actual_stimuli_sha",
  "rows_per_model_layer": 400,
  "source_patch_layer": 8,
  "runs": ["Qwen_L21", "Qwen_L23", "Qwen_L24", "Qwen_L26", "Qwen_L27", "Gemma_L29", "Gemma_L35"]
}
EOF
