#!/usr/bin/env bash
set -euo pipefail

CODE_ROOT="${CODE_ROOT:-/home/ubuntu/CoT-Native-thinking-v5/code/Realistic_CoT_NiaH_Count}"
RUN_ROOT="${RUN_ROOT:-/home/ubuntu/CoT-Native-thinking-v5/runs/v5_native_hybrid_supplement_20260820}"
PYTHON="${PYTHON:-/home/ubuntu/CoT-Native-thinking-v5/venv/bin/python}"
CACHE_DIR="${CACHE_DIR:-/home/ubuntu/CoT-Native-thinking-v5/cache/huggingface}"
GPU_COUNT="${GPU_COUNT:-8}"
CONFIG="$CODE_ROOT/configs/realistic_niah_v5.json"
SPEC="$CODE_ROOT/configs/realistic_niah_v5_native_hybrid_localizer_p0_ablation_frozen.json"
QWEN_GENERATIONS="${QWEN_GENERATIONS:-$CODE_ROOT/work/v5_trace_parser_v2/Qwen3-8B_generations_reparsed.jsonl}"
GEMMA_GENERATIONS="${GEMMA_GENERATIONS:-$CODE_ROOT/work/v5_trace_parser_v2/Gemma4-E4B_generations_reparsed.jsonl}"
QWEN_ROUTING="$CODE_ROOT/configs/realistic_niah_v5_qwen_hybrid_localizer_p0_ablation_routes_frozen.json"
GEMMA_ROUTING="$CODE_ROOT/configs/realistic_niah_v5_gemma_hybrid_localizer_p0_ablation_routes_frozen.json"
LOG="$RUN_ROOT/logs/native_hybrid_supplement_8gpu_supervisor.log"
JOBS="$RUN_ROOT/supplement_behavior_jobs.jsonl"
SIDECAR_JOBS="$RUN_ROOT/sidecar_jobs.jsonl"

QWEN_GRAMMARS=(
  adjacent_rank_before_city
  same_unit_rank_before_city
  structural_explicit_rank_before_city
)
QWEN_DOSES=(32 64 80 96 112 128)
GEMMA_GRAMMARS=(adjacent_rank_before_city same_unit_rank_before_city)
GEMMA_DOSES=(1 2 4 6 8)

mkdir -p "$RUN_ROOT/logs" "$RUN_ROOT/locks" "$CACHE_DIR"
exec 9>"$RUN_ROOT/locks/native_hybrid_supplement_8gpu.lock"
if ! flock -n 9; then
  echo "another native hybrid supplement supervisor owns the lock" >&2
  exit 75
fi
exec > >(tee -a "$LOG") 2>&1
cd "$CODE_ROOT"

"$PYTHON" scripts/validate_v5_native_hybrid_localizer_spec.py \
  --spec "$SPEC" \
  --config "$CONFIG" \
  --qwen-routing "$QWEN_ROUTING" \
  --gemma-routing "$GEMMA_ROUTING"
if test "$GPU_COUNT" -lt 1; then
  echo "GPU_COUNT must be positive" >&2
  exit 64
fi

model_root() { echo "$RUN_ROOT/$1"; }
source_dir() { echo "$(model_root "$1")/source_writes/post_marker"; }
plan_dir() { echo "$(model_root "$1")/plans/$2/post_marker/k$3"; }

wait_group() {
  local label="$1"
  shift
  local failed=0 pid
  for pid in "$@"; do
    if ! wait "$pid"; then failed=1; fi
  done
  if test "$failed" -ne 0; then
    echo "GROUP_FAILED label=$label; inspect component logs" >&2
    exit 1
  fi
  echo "GROUP_COMPLETE label=$label utc=$(date -u +%FT%TZ)"
}

run_source() {
  local gpu="$1" model="$2" generations="$3"
  local output log
  output="$(source_dir "$model")"
  log="$(model_root "$model")/logs/source_post_marker.log"
  mkdir -p "$(dirname "$log")"
  echo "SOURCE_START gpu=$gpu model=$model role=post_marker"
  CUDA_VISIBLE_DEVICES="$gpu" HF_HOME="$CACHE_DIR" TOKENIZERS_PARALLELISM=false \
    "$PYTHON" scripts/run_realistic_niah_v5.py causal-source-writes \
      --config "$CONFIG" \
      --model "$model" \
      --cache-dir "$CACHE_DIR" \
      --device-map auto \
      --torch-dtype bfloat16 \
      --attention-backend sdpa \
      --generations "$generations" \
      --output "$output" \
      --anchor-role post_marker \
      --include-secondary >"$log" 2>&1
  echo "SOURCE_COMPLETE gpu=$gpu model=$model role=post_marker"
}

echo "STAGE new_post_marker_source_writes utc=$(date -u +%FT%TZ)"
run_source 0 Qwen3-8B "$QWEN_GENERATIONS" & qwen_source_pid=$!
run_source 1 Gemma4-E4B "$GEMMA_GENERATIONS" & gemma_source_pid=$!
wait_group post_marker_source_writes "$qwen_source_pid" "$gemma_source_pid"

build_plan() {
  local model="$1" grammar="$2" k="$3" policy="$4"
  local output log matching status
  output="$(plan_dir "$model" "$grammar" "$k")"
  log="$(model_root "$model")/logs/plan_${grammar}_post_marker_k${k}.log"
  mkdir -p "$output" "$(dirname "$log")"
  if test -f "$output/causal_plan_audit.json" \
    && grep -q "\"registered_bank_size\": $k" "$output/causal_plan_audit.json"; then
    echo "PLAN_REUSE model=$model grammar=$grammar K=$k"
    return
  fi
  matching=global
  if test "$policy" = prefer_layer; then matching=layer_matched; fi
  echo "PLAN_START model=$model grammar=$grammar K=$k matching=$matching"
  set +e
  "$PYTHON" scripts/run_realistic_niah_v5.py causal-plan \
    --config "$CONFIG" \
    --source-writes "$(source_dir "$model")" \
    --output "$output" \
    --bank-size "$k" \
    --anchor-role post_marker \
    --target-grammar-class "$grammar" \
    --selection-metric target_source_attention_mass \
    --selection-eligibility-scope local \
    --selection-aggregation seed_event_mean \
    --random-control-matching "$matching" \
    --full-panel-plan >"$log" 2>&1
  status=$?
  set -e
  if test "$status" -eq 0; then
    echo "PLAN_COMPLETE model=$model grammar=$grammar K=$k matching=$matching"
    return
  fi
  if test "$policy" != prefer_layer; then
    echo "PLAN_FAILED model=$model grammar=$grammar K=$k log=$log" >&2
    exit "$status"
  fi
  echo "PLAN_FALLBACK model=$model grammar=$grammar K=$k to=global"
  "$PYTHON" scripts/run_realistic_niah_v5.py causal-plan \
    --config "$CONFIG" \
    --source-writes "$(source_dir "$model")" \
    --output "$output" \
    --bank-size "$k" \
    --anchor-role post_marker \
    --target-grammar-class "$grammar" \
    --selection-metric target_source_attention_mass \
    --selection-eligibility-scope local \
    --selection-aggregation seed_event_mean \
    --random-control-matching global \
    --full-panel-plan >>"$log" 2>&1
  echo "PLAN_COMPLETE model=$model grammar=$grammar K=$k matching=global_fallback"
}

echo "STAGE new_post_marker_plan_matrix utc=$(date -u +%FT%TZ)"
for grammar in "${QWEN_GRAMMARS[@]}"; do
  for k in "${QWEN_DOSES[@]}"; do
    if test "$k" -eq 128; then
      build_plan Qwen3-8B "$grammar" "$k" global
    else
      build_plan Qwen3-8B "$grammar" "$k" prefer_layer
    fi
  done
done
for grammar in "${GEMMA_GRAMMARS[@]}"; do
  for k in "${GEMMA_DOSES[@]}"; do
    build_plan Gemma4-E4B "$grammar" "$k" prefer_layer
  done
done

echo "STAGE reused_p0_ranking_sidecar_plan utc=$(date -u +%FT%TZ)"
SIDECAR_ROOT="$(model_root Qwen3-8B)/diagnostics/adjacent_rank_before_city_p0_score_post_marker_layer_k112"
if ! test -f "$SIDECAR_ROOT/plan/causal_plan_audit.json"; then
  "$PYTHON" scripts/build_v5_native_layer_profile_control_plan.py \
    --p0-ranking "$CODE_ROOT/work/v5_qwen_p0_grammar_head_bank_overlap_v1/ranking_adjacent_rank_before_city.csv" \
    --p2-plan-dir "$(plan_dir Qwen3-8B adjacent_rank_before_city 112)" \
    --output "$SIDECAR_ROOT/plan" \
    --random-control-matching layer_matched \
    --random-repeats 3
fi

freeze_registry() {
  local gpu="$1" model="$2" generations="$3" routing="$4" plan="$5"
  local root output log
  root="$(model_root "$model")"
  output="$root/registries/all"
  log="$root/logs/freeze_registry.log"
  if test -f "$output/selected_anchor_registry.jsonl"; then
    echo "REGISTRY_REUSE model=$model"
    return
  fi
  echo "REGISTRY_RECONSTRUCT_START gpu=$gpu model=$model no_behavior_forward=1"
  CUDA_VISIBLE_DEVICES="$gpu" HF_HOME="$CACHE_DIR" TOKENIZERS_PARALLELISM=false \
    "$PYTHON" scripts/run_realistic_niah_v5.py causal-heads-behavior \
      --config "$CONFIG" \
      --model "$model" \
      --cache-dir "$CACHE_DIR" \
      --device-map auto \
      --torch-dtype bfloat16 \
      --attention-backend sdpa \
      --generations "$generations" \
      --plan "$plan" \
      --output "$output" \
      --anchor-routing "$routing" \
      --behavior-all-routed-grammars \
      --allow-selection-scope-bank-transfer \
      --allow-selection-intervention-site-decoupling \
      --evaluation-split all \
      --conditions clean \
      --include-secondary \
      --limit 300 \
      --anchor-sampling prompt_balanced \
      --freeze-anchor-registry-only \
      --max-new-tokens 256 \
      --decode-head-ablation-steps -1 >"$log" 2>&1
  echo "REGISTRY_RECONSTRUCT_COMPLETE gpu=$gpu model=$model"
}

QWEN_REGISTRY_PLAN="$(plan_dir Qwen3-8B adjacent_rank_before_city 128)/retrieval_anchor_bank_plan.csv"
GEMMA_REGISTRY_PLAN="$(plan_dir Gemma4-E4B adjacent_rank_before_city 8)/retrieval_anchor_bank_plan.csv"
echo "STAGE deterministic_registry_reconstruction utc=$(date -u +%FT%TZ)"
freeze_registry 0 Qwen3-8B "$QWEN_GENERATIONS" "$QWEN_ROUTING" "$QWEN_REGISTRY_PLAN" & qwen_registry_pid=$!
freeze_registry 1 Gemma4-E4B "$GEMMA_GENERATIONS" "$GEMMA_ROUTING" "$GEMMA_REGISTRY_PLAN" & gemma_registry_pid=$!
wait_group registry_reconstruction "$qwen_registry_pid" "$gemma_registry_pid"
for model in Qwen3-8B Gemma4-E4B; do
  "$PYTHON" scripts/partition_v5_native_hybrid_anchor_registry.py \
    --spec "$SPEC" \
    --model "$model" \
    --registry "$(model_root "$model")/registries/all/selected_anchor_registry.jsonl" \
    --output "$(model_root "$model")/registries/by_grammar"
done

"$PYTHON" scripts/build_v5_native_hybrid_behavior_jobs.py \
  --spec "$SPEC" \
  --config "$CONFIG" \
  --run-root "$RUN_ROOT" \
  --output "$JOBS" \
  --sidecar-output "$SIDECAR_JOBS" \
  --supplement-only

echo "STAGE supplement_behavior_grid workers=$GPU_COUNT utc=$(date -u +%FT%TZ)"
worker_pids=()
for ((gpu=0; gpu<GPU_COUNT; gpu++)); do
  CUDA_VISIBLE_DEVICES="$gpu" \
    "$PYTHON" scripts/run_v5_native_hybrid_behavior_worker.py \
      --jobs "$JOBS" \
      --worker-index "$gpu" \
      --worker-count "$GPU_COUNT" \
      --code-root "$CODE_ROOT" \
      --python "$PYTHON" \
      --config "$CONFIG" \
      --cache-dir "$CACHE_DIR" \
      --qwen-generations "$QWEN_GENERATIONS" \
      --gemma-generations "$GEMMA_GENERATIONS" \
      --qwen-routing "$QWEN_ROUTING" \
      --gemma-routing "$GEMMA_ROUTING" >"$RUN_ROOT/logs/behavior_worker_gpu${gpu}.log" 2>&1 &
  worker_pids+=("$!")
done
wait_group supplement_behavior_grid "${worker_pids[@]}"

echo "STAGE sidecar_behavior gpu=0 utc=$(date -u +%FT%TZ)"
CUDA_VISIBLE_DEVICES=0 \
  "$PYTHON" scripts/run_v5_native_hybrid_behavior_worker.py \
    --jobs "$SIDECAR_JOBS" \
    --worker-index 0 \
    --worker-count 1 \
    --code-root "$CODE_ROOT" \
    --python "$PYTHON" \
    --config "$CONFIG" \
    --cache-dir "$CACHE_DIR" \
    --qwen-generations "$QWEN_GENERATIONS" \
    --gemma-generations "$GEMMA_GENERATIONS" \
    --qwen-routing "$QWEN_ROUTING" \
    --gemma-routing "$GEMMA_ROUTING" >"$RUN_ROOT/logs/sidecar_worker_gpu0.log" 2>&1

echo "STAGE validate_new_and_merge_frozen_p0 utc=$(date -u +%FT%TZ)"
for model in Qwen3-8B Gemma4-E4B; do
  "$PYTHON" scripts/validate_and_merge_v5_native_hybrid_supplement.py \
    --spec "$SPEC" \
    --config "$CONFIG" \
    --code-root "$CODE_ROOT" \
    --run-root "$RUN_ROOT" \
    --model "$model" \
    --jobs "$JOBS" \
    --sidecar-jobs "$SIDECAR_JOBS" \
    --output "$(model_root "$model")/analysis_hybrid_supplement_registered_v1"
done

"$PYTHON" - "$RUN_ROOT" "$SPEC" <<'PY'
import datetime
import hashlib
import json
import pathlib
import sys

root = pathlib.Path(sys.argv[1])
spec = pathlib.Path(sys.argv[2])
models = {}
for model in ("Qwen3-8B", "Gemma4-E4B"):
    analysis = root / model / "analysis_hybrid_supplement_registered_v1"
    supplement = analysis / "supplement_complete.json"
    merged = analysis / "hybrid_dose_grid_complete.json"
    assert json.loads(supplement.read_text(encoding="utf-8"))["status"] == "PASS"
    models[model] = {
        "supplement": str(supplement),
        "supplement_sha256": hashlib.sha256(supplement.read_bytes()).hexdigest(),
        "merged_dose": str(merged),
        "merged_dose_sha256": hashlib.sha256(merged.read_bytes()).hexdigest(),
    }
payload = {
    "schema_version": "realistic_niah_v5_native_hybrid_supplement_8gpu_v1",
    "status": "PASS",
    "completed_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
    "spec_sha256": hashlib.sha256(spec.read_bytes()).hexdigest(),
    "execution": "new_P2_ranked_grammars_only_plus_frozen_P0_result_reuse",
    "models": models,
}
path = root / "native_hybrid_supplement_8gpu_complete.json"
path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
print(json.dumps(payload, sort_keys=True))
PY

echo "ALL_COMPLETE utc=$(date -u +%FT%TZ) completion=$RUN_ROOT/native_hybrid_supplement_8gpu_complete.json"
