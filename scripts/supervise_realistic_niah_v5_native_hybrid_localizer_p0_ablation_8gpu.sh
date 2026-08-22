#!/usr/bin/env bash
set -euo pipefail

# Safe default: the completed P0 grammar cells are frozen and reused.  The
# expensive full-panel implementation below remains available only for an
# explicit forensic rerun.
if test "${ALLOW_FULL_RERUN:-0}" != "1"; then
  exec "$(dirname "$0")/supervise_realistic_niah_v5_native_hybrid_supplement_8gpu.sh" "$@"
fi

CODE_ROOT="${CODE_ROOT:-/home/ubuntu/CoT-Native-thinking-v5/code/Realistic_CoT_NiaH_Count}"
RUN_ROOT="${RUN_ROOT:-/home/ubuntu/CoT-Native-thinking-v5/runs/v5_native_hybrid_localizer_p0_20260820}"
PYTHON="${PYTHON:-/home/ubuntu/CoT-Native-thinking-v5/venv/bin/python}"
CACHE_DIR="${CACHE_DIR:-/home/ubuntu/CoT-Native-thinking-v5/cache/huggingface}"
GPU_COUNT="${GPU_COUNT:-8}"
CONFIG="$CODE_ROOT/configs/realistic_niah_v5.json"
SPEC="$CODE_ROOT/configs/realistic_niah_v5_native_hybrid_localizer_p0_ablation_frozen.json"
QWEN_GENERATIONS="${QWEN_GENERATIONS:-$CODE_ROOT/work/v5_trace_parser_v2/Qwen3-8B_generations_reparsed.jsonl}"
GEMMA_GENERATIONS="${GEMMA_GENERATIONS:-$CODE_ROOT/work/v5_trace_parser_v2/Gemma4-E4B_generations_reparsed.jsonl}"
QWEN_ROUTING="$CODE_ROOT/configs/realistic_niah_v5_qwen_hybrid_localizer_p0_ablation_routes_frozen.json"
GEMMA_ROUTING="$CODE_ROOT/configs/realistic_niah_v5_gemma_hybrid_localizer_p0_ablation_routes_frozen.json"
LOG="$RUN_ROOT/logs/native_hybrid_localizer_p0_8gpu_supervisor.log"
JOBS="$RUN_ROOT/behavior_jobs.jsonl"
SIDECAR_JOBS="$RUN_ROOT/sidecar_jobs.jsonl"

QWEN_GRAMMARS=(
  adjacent_rank_after_city
  adjacent_rank_before_city
  same_unit_rank_before_city
  structural_unmarked
  structural_invariant_bullet
  evidence_sequence_unranked
  structural_explicit_rank_before_city
)
QWEN_DOSES=(32 64 80 96 112 128)
GEMMA_GRAMMARS=(
  adjacent_rank_after_city
  adjacent_rank_before_city
  same_unit_rank_after_city
  same_unit_rank_before_city
  structural_invariant_bullet
)
GEMMA_DOSES=(1 2 4 6 8)

mkdir -p "$RUN_ROOT/logs" "$RUN_ROOT/locks" "$CACHE_DIR"
exec 9>"$RUN_ROOT/locks/native_hybrid_localizer_p0_8gpu.lock"
if ! flock -n 9; then
  echo "another hybrid-localizer P0 supervisor owns the lock" >&2
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

qwen_role() {
  case "$1" in
    adjacent_rank_before_city|same_unit_rank_before_city|structural_explicit_rank_before_city)
      echo post_marker ;;
    *) echo p0_item_end ;;
  esac
}

gemma_role() {
  case "$1" in
    adjacent_rank_before_city|same_unit_rank_before_city)
      echo post_marker ;;
    *) echo p0_item_end ;;
  esac
}

model_root() {
  echo "$RUN_ROOT/$1"
}

source_dir() {
  echo "$(model_root "$1")/source_writes/$2"
}

plan_dir() {
  echo "$(model_root "$1")/plans/$2/$3/k$4"
}

run_source() {
  local gpu="$1" model="$2" generations="$3" role="$4"
  local output source_log
  output="$(source_dir "$model" "$role")"
  source_log="$(model_root "$model")/logs/source_${role}.log"
  mkdir -p "$(dirname "$source_log")"
  echo "SOURCE_START gpu=$gpu model=$model role=$role utc=$(date -u +%FT%TZ)"
  CUDA_VISIBLE_DEVICES="$gpu" \
    HF_HOME="$CACHE_DIR" TOKENIZERS_PARALLELISM=false \
    "$PYTHON" scripts/run_realistic_niah_v5.py causal-source-writes \
      --config "$CONFIG" \
      --model "$model" \
      --cache-dir "$CACHE_DIR" \
      --device-map auto \
      --torch-dtype bfloat16 \
      --attention-backend sdpa \
      --generations "$generations" \
      --output "$output" \
      --anchor-role "$role" \
      --include-secondary >"$source_log" 2>&1
  echo "SOURCE_COMPLETE gpu=$gpu model=$model role=$role utc=$(date -u +%FT%TZ)"
}

wait_group() {
  local label="$1"
  shift
  local failed=0 pid
  for pid in "$@"; do
    if ! wait "$pid"; then
      failed=1
    fi
  done
  if test "$failed" -ne 0; then
    echo "GROUP_FAILED label=$label; inspect component logs" >&2
    exit 1
  fi
  echo "GROUP_COMPLETE label=$label utc=$(date -u +%FT%TZ)"
}

echo "STAGE source_writes utc=$(date -u +%FT%TZ)"
source_pids=()
run_source 0 Qwen3-8B "$QWEN_GENERATIONS" p0_item_end & source_pids+=("$!")
run_source 1 Qwen3-8B "$QWEN_GENERATIONS" post_marker & source_pids+=("$!")
run_source 2 Gemma4-E4B "$GEMMA_GENERATIONS" p0_item_end & source_pids+=("$!")
run_source 3 Gemma4-E4B "$GEMMA_GENERATIONS" post_marker & source_pids+=("$!")
wait_group source_writes "${source_pids[@]}"

build_plan() {
  local model="$1" grammar="$2" role="$3" k="$4" policy="$5"
  local output source plan_log
  output="$(plan_dir "$model" "$grammar" "$role" "$k")"
  source="$(source_dir "$model" "$role")"
  plan_log="$(model_root "$model")/logs/plan_${grammar}_${role}_k${k}.log"
  mkdir -p "$output" "$(dirname "$plan_log")"
  if test -f "$output/causal_plan_audit.json" \
    && grep -q "\"registered_bank_size\": $k" "$output/causal_plan_audit.json"; then
    echo "PLAN_REUSE model=$model grammar=$grammar role=$role K=$k"
    return
  fi
  local matching=global
  if test "$policy" = "prefer_layer"; then
    matching=layer_matched
  fi
  echo "PLAN_START model=$model grammar=$grammar role=$role K=$k matching=$matching"
  set +e
  "$PYTHON" scripts/run_realistic_niah_v5.py causal-plan \
    --config "$CONFIG" \
    --source-writes "$source" \
    --output "$output" \
    --bank-size "$k" \
    --anchor-role "$role" \
    --target-grammar-class "$grammar" \
    --selection-metric target_source_attention_mass \
    --selection-eligibility-scope local \
    --selection-aggregation seed_event_mean \
    --random-control-matching "$matching" \
    --full-panel-plan >"$plan_log" 2>&1
  local status=$?
  set -e
  if test "$status" -eq 0; then
    echo "PLAN_COMPLETE model=$model grammar=$grammar role=$role K=$k matching=$matching"
    return
  fi
  if test "$policy" != "prefer_layer"; then
    echo "PLAN_FAILED model=$model grammar=$grammar role=$role K=$k; inspect $plan_log" >&2
    exit "$status"
  fi
  echo "PLAN_FALLBACK model=$model grammar=$grammar role=$role K=$k from=layer_matched to=global"
  "$PYTHON" scripts/run_realistic_niah_v5.py causal-plan \
    --config "$CONFIG" \
    --source-writes "$source" \
    --output "$output" \
    --bank-size "$k" \
    --anchor-role "$role" \
    --target-grammar-class "$grammar" \
    --selection-metric target_source_attention_mass \
    --selection-eligibility-scope local \
    --selection-aggregation seed_event_mean \
    --random-control-matching global \
    --full-panel-plan >>"$plan_log" 2>&1
  echo "PLAN_COMPLETE model=$model grammar=$grammar role=$role K=$k matching=global_fallback"
}

echo "STAGE frozen_plan_matrix utc=$(date -u +%FT%TZ)"
for grammar in "${QWEN_GRAMMARS[@]}"; do
  role="$(qwen_role "$grammar")"
  for k in "${QWEN_DOSES[@]}"; do
    if test "$k" -eq 128; then
      build_plan Qwen3-8B "$grammar" "$role" "$k" global
    else
      build_plan Qwen3-8B "$grammar" "$role" "$k" prefer_layer
    fi
  done
done
for grammar in "${GEMMA_GRAMMARS[@]}"; do
  role="$(gemma_role "$grammar")"
  for k in "${GEMMA_DOSES[@]}"; do
    build_plan Gemma4-E4B "$grammar" "$role" "$k" prefer_layer
  done
done

echo "STAGE layer_profile_sidecar_plan utc=$(date -u +%FT%TZ)"
SIDECAR_ROOT="$(model_root Qwen3-8B)/diagnostics/adjacent_rank_before_city_p0_score_post_marker_layer_k112"
SIDECAR_P0_PLAN="$SIDECAR_ROOT/p0_reference_plan"
if ! test -f "$SIDECAR_P0_PLAN/causal_plan_audit.json"; then
  "$PYTHON" scripts/run_realistic_niah_v5.py causal-plan \
    --config "$CONFIG" \
    --source-writes "$(source_dir Qwen3-8B p0_item_end)" \
    --output "$SIDECAR_P0_PLAN" \
    --bank-size 112 \
    --anchor-role p0_item_end \
    --target-grammar-class adjacent_rank_before_city \
    --selection-metric target_source_attention_mass \
    --selection-eligibility-scope local \
    --selection-aggregation seed_event_mean \
    --selected-only-smoke \
    --full-panel-plan
fi
if ! test -f "$SIDECAR_ROOT/plan/causal_plan_audit.json"; then
  "$PYTHON" scripts/build_v5_native_layer_profile_control_plan.py \
    --p0-plan-dir "$SIDECAR_P0_PLAN" \
    --p2-plan-dir "$(plan_dir Qwen3-8B adjacent_rank_before_city post_marker 112)" \
    --output "$SIDECAR_ROOT/plan" \
    --random-control-matching layer_matched \
    --random-repeats 3
fi

freeze_registry() {
  local gpu="$1" model="$2" generations="$3" routing="$4" pooled_plan="$5"
  local root output registry_log
  root="$(model_root "$model")"
  output="$root/registries/all"
  registry_log="$root/logs/freeze_registry.log"
  mkdir -p "$root/logs"
  if test -f "$output/selected_anchor_registry.jsonl"; then
    echo "REGISTRY_REUSE model=$model"
    return
  fi
  echo "REGISTRY_START gpu=$gpu model=$model"
  CUDA_VISIBLE_DEVICES="$gpu" \
    HF_HOME="$CACHE_DIR" TOKENIZERS_PARALLELISM=false \
    "$PYTHON" scripts/run_realistic_niah_v5.py causal-heads-behavior \
      --config "$CONFIG" \
      --model "$model" \
      --cache-dir "$CACHE_DIR" \
      --device-map auto \
      --torch-dtype bfloat16 \
      --attention-backend sdpa \
      --generations "$generations" \
      --plan "$pooled_plan" \
      --output "$output" \
      --anchor-routing "$routing" \
      --behavior-all-routed-grammars \
      --allow-selection-scope-bank-transfer \
      --evaluation-split all \
      --conditions clean \
      --include-secondary \
      --limit 300 \
      --anchor-sampling prompt_balanced \
      --freeze-anchor-registry-only \
      --max-new-tokens 256 \
      --decode-head-ablation-steps -1 >"$registry_log" 2>&1
  echo "REGISTRY_COMPLETE gpu=$gpu model=$model"
}

QWEN_POOLED_PLAN="$(plan_dir Qwen3-8B adjacent_rank_after_city p0_item_end 128)/retrieval_anchor_bank_plan.csv"
GEMMA_POOLED_PLAN="$(plan_dir Gemma4-E4B adjacent_rank_after_city p0_item_end 8)/retrieval_anchor_bank_plan.csv"
echo "STAGE registry_freeze utc=$(date -u +%FT%TZ)"
freeze_registry 0 Qwen3-8B "$QWEN_GENERATIONS" "$QWEN_ROUTING" "$QWEN_POOLED_PLAN" & registry_qwen_pid=$!
freeze_registry 1 Gemma4-E4B "$GEMMA_GENERATIONS" "$GEMMA_ROUTING" "$GEMMA_POOLED_PLAN" & registry_gemma_pid=$!
wait_group registry_freeze "$registry_qwen_pid" "$registry_gemma_pid"

for model in Qwen3-8B Gemma4-E4B; do
  "$PYTHON" scripts/partition_v5_native_hybrid_anchor_registry.py \
    --spec "$SPEC" \
    --model "$model" \
    --registry "$(model_root "$model")/registries/all/selected_anchor_registry.jsonl" \
    --output "$(model_root "$model")/registries/by_grammar"
done

run_clean() {
  local gpu="$1" model="$2" generations="$3" routing="$4" plan="$5"
  local root output clean_log registry
  root="$(model_root "$model")"
  output="$root/behaviors/clean_full"
  clean_log="$root/logs/clean_full.log"
  registry="$root/registries/all/selected_anchor_registry.jsonl"
  if test -f "$output/manifest.json" \
    && "$PYTHON" -c 'import json,sys; x=json.load(open(sys.argv[1])); raise SystemExit(0 if x.get("completed_shards")==x.get("scheduled_anchor_condition_trials") and x.get("completed_shards",0)>0 else 1)' "$output/manifest.json"; then
    echo "CLEAN_REUSE model=$model"
    return
  fi
  echo "CLEAN_START gpu=$gpu model=$model"
  CUDA_VISIBLE_DEVICES="$gpu" \
    HF_HOME="$CACHE_DIR" TOKENIZERS_PARALLELISM=false \
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
      --evaluation-split all \
      --conditions clean \
      --include-secondary \
      --limit 300 \
      --anchor-sampling prompt_balanced \
      --anchor-registry-input "$registry" \
      --max-new-tokens 256 \
      --decode-head-ablation-steps -1 >"$clean_log" 2>&1
  echo "CLEAN_COMPLETE gpu=$gpu model=$model"
}

echo "STAGE clean_baselines utc=$(date -u +%FT%TZ)"
run_clean 0 Qwen3-8B "$QWEN_GENERATIONS" "$QWEN_ROUTING" "$QWEN_POOLED_PLAN" & clean_qwen_pid=$!
run_clean 1 Gemma4-E4B "$GEMMA_GENERATIONS" "$GEMMA_ROUTING" "$GEMMA_POOLED_PLAN" & clean_gemma_pid=$!
wait_group clean_baselines "$clean_qwen_pid" "$clean_gemma_pid"

"$PYTHON" scripts/build_v5_native_hybrid_behavior_jobs.py \
  --spec "$SPEC" \
  --config "$CONFIG" \
  --run-root "$RUN_ROOT" \
  --output "$JOBS" \
  --sidecar-output "$SIDECAR_JOBS"

echo "STAGE behavior_grid workers=$GPU_COUNT utc=$(date -u +%FT%TZ)"
worker_pids=()
for ((gpu=0; gpu<GPU_COUNT; gpu++)); do
  worker_log="$RUN_ROOT/logs/behavior_worker_gpu${gpu}.log"
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
      --gemma-routing "$GEMMA_ROUTING" >"$worker_log" 2>&1 &
  worker_pids+=("$!")
done
wait_group behavior_grid "${worker_pids[@]}"

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

echo "STAGE validation utc=$(date -u +%FT%TZ)"
for model in Qwen3-8B Gemma4-E4B; do
  "$PYTHON" scripts/validate_v5_native_hybrid_localizer_p0_ablation.py \
    --spec "$SPEC" \
    --config "$CONFIG" \
    --run-root "$RUN_ROOT" \
    --model "$model" \
    --jobs "$JOBS" \
    --sidecar-jobs "$SIDECAR_JOBS" \
    --output "$(model_root "$model")/analysis_hybrid_localizer_p0_registered_v1"
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
    path = root / model / "analysis_hybrid_localizer_p0_registered_v1" / "hybrid_localizer_p0_ablation_complete.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["status"] == "PASS", (model, payload.get("status"))
    models[model] = {
        "completion": str(path),
        "completion_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "registry_rows": payload["registry_rows"],
        "registry_sha256": payload["registry_sha256"],
    }
complete = {
    "schema_version": "realistic_niah_v5_native_hybrid_localizer_p0_8gpu_v1",
    "status": "PASS",
    "completed_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
    "spec_sha256": hashlib.sha256(spec.read_bytes()).hexdigest(),
    "models": models,
}
path = root / "native_hybrid_localizer_p0_8gpu_complete.json"
path.write_text(json.dumps(complete, indent=2, sort_keys=True) + "\n", encoding="utf-8")
print(json.dumps(complete, sort_keys=True))
PY

echo "ALL_COMPLETE utc=$(date -u +%FT%TZ) completion=$RUN_ROOT/native_hybrid_localizer_p0_8gpu_complete.json"
