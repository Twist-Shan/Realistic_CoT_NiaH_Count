#!/usr/bin/env bash
set -euo pipefail

repo="/home/ubuntu/Realistic_CoT_NiaH_Count"
root="/home/ubuntu/runs/nonthinking_v445_8gpu_20260813"
manifest="$repo/configs/realistic_niah_v4_4_5_8gpu_shards.json"
stimuli="$root/dataset/canonical_run_20260731_v4_numeric_presentation_v3_stimuli.jsonl"
cache="/home/ubuntu/hf-cache"
expected_sha="da4dd86142eb8a07f9a7e53497efd3375184c8e68367d4db994370fcb331f090"

mkdir -p "$root/logs" "$root/locks" "$root/shards"
cd "$repo"
actual_sha="$(sha256sum "$stimuli" | cut -d' ' -f1)"
if [[ "$actual_sha" != "$expected_sha" ]]; then
  echo "Stimulus SHA audit failed: $actual_sha" >&2
  exit 1
fi
PYTHONPATH=src .venv/bin/python \
  scripts/audit_realistic_niah_v4_4_5_8gpu_prelaunch.py \
  --manifest "$manifest" \
  --allow-resume >"$root/prelaunch_audit.json"

launch_worker() {
  local label="$1" gpu="$2" model="$3" seeds="$4" layers="$5"
  local output="$root/shards/$label"
  mkdir -p "$output"
  (
    flock -n 9 || exit 75
    export CUDA_VISIBLE_DEVICES="$gpu"
    export PYTHONPATH=src
    exec .venv/bin/python scripts/run_realistic_niah_v4_4_5_span_restoration.py \
      --model "$model" \
      --stimuli "$stimuli" \
      --stimuli-config configs/realistic_niah_v4.json \
      --experiment-config configs/realistic_niah_v4_4_5_span_restoration_canonical.json \
      --output-dir "$output" \
      --cache-dir "$cache" \
      --device-map auto \
      --seeds "$seeds" \
      --counts "1,2,3,4,5,6,7,8,9,10" \
      --layers "$layers" \
      --patch-kinds "needle_endpoint,needle_full,ordinary_full" \
      --skip-cache-equivalence-audit \
      --reuse-prefill-for-generation
  ) 9>"$root/locks/$label.lock" >"$root/logs/$label.log" 2>&1 &
  worker_pids+=("$!")
  worker_labels+=("$label")
}

qwen_layers="$(seq -s, 0 35)"
gemma_layers="$(seq -s, 0 41)"
worker_pids=()
worker_labels=()
launch_worker qwen_0 0 Qwen3-8B "$(seq -s, 1234 1248)" "$qwen_layers"
launch_worker qwen_1 1 Qwen3-8B "$(seq -s, 1249 1263)" "$qwen_layers"
launch_worker gemma_0 2 Gemma4-E4B "$(seq -s, 1234 1238)" "$gemma_layers"
launch_worker gemma_1 3 Gemma4-E4B "$(seq -s, 1239 1243)" "$gemma_layers"
launch_worker gemma_2 4 Gemma4-E4B "$(seq -s, 1244 1248)" "$gemma_layers"
launch_worker gemma_3 5 Gemma4-E4B "$(seq -s, 1249 1253)" "$gemma_layers"
launch_worker gemma_4 6 Gemma4-E4B "$(seq -s, 1254 1258)" "$gemma_layers"
launch_worker gemma_5 7 Gemma4-E4B "$(seq -s, 1259 1263)" "$gemma_layers"

failed=0
for index in "${!worker_pids[@]}"; do
  if ! wait "${worker_pids[$index]}"; then
    echo "Worker ${worker_labels[$index]} failed" >&2
    failed=1
  fi
done
if [[ "$failed" -ne 0 ]]; then
  exit 1
fi

for label in qwen_0 qwen_1; do
  rows="$(wc -l < "$root/shards/$label/Qwen3-8B/detail.jsonl")"
  [[ "$rows" -eq 16650 ]] || { echo "$label rows=$rows" >&2; exit 1; }
done
for label in gemma_0 gemma_1 gemma_2 gemma_3 gemma_4 gemma_5; do
  rows="$(wc -l < "$root/shards/$label/Gemma4-E4B/detail.jsonl")"
  [[ "$rows" -eq 6450 ]] || { echo "$label rows=$rows" >&2; exit 1; }
done

cat >"$root/dense_8gpu_complete.json" <<EOF
{
  "status": "complete",
  "stimulus_sha256": "$actual_sha",
  "qwen_rows": 33300,
  "gemma_rows": 38700,
  "shards": 8
}
EOF
