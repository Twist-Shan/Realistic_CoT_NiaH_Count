#!/usr/bin/env bash
set -Eeuo pipefail

usage() {
  cat <<'EOF'
Usage:
  launch_realistic_niah_v4_4_correct_interventions_4x4.sh \
    RUN_ROOT VENV_PYTHON HF_CACHE QWEN_SOURCE_RUN GEMMA_SOURCE_RUN \
    ABLATION_CONFIRMATION_SOURCE_RUN

Formal eight-GPU launcher for the V4.4 correct-intervention extension. It uses
GPUs 0--3 for four independent Qwen workers and GPUs 4--7 for four independent
Gemma workers. Each model first performs its baseline-only stopping scan on one
GPU, freezes a deterministic sorted-round-robin work plan, then launches four
single-GPU workers into isolated directories. A CPU merge verifies disjoint and
exhaustive coverage before the ordinary strict campaign audit runs.

The launcher requires at least eight visible GPUs with at least 75 GiB each.
It never lets two workers write the same shard path. Rerunning the same command
reuses completed baseline and intervention shards after design/hash checks.
EOF
}

if [[ $# -ne 6 ]]; then
  usage >&2
  exit 2
fi

RUN_ROOT=$1
VENV_PYTHON=$2
HF_CACHE=$3
QWEN_SOURCE_RUN=$4
GEMMA_SOURCE_RUN=$5
ABLATION_SOURCE_RUN=$6
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd -- "$SCRIPT_DIR/.." && pwd)
BASE_CONFIG="$REPO_ROOT/configs/realistic_niah_v4.json"
CAUSAL_CONFIG="$REPO_ROOT/configs/realistic_niah_v4_causal_v2.json"
DEFINITION="$REPO_ROOT/configs/realistic_niah_v4_4_correct_interventions.json"
PARALLEL_RUNNER="$REPO_ROOT/scripts/run_realistic_niah_v4_4_correct_interventions_parallel.py"
AUDITOR="$REPO_ROOT/scripts/audit_realistic_niah_v4_4_correct_interventions.py"
DATASET_ROOT="$RUN_ROOT/dataset/correct_interventions"
STIMULI="$DATASET_ROOT/stimuli_v4_4_causal_v2.jsonl"
LOG_ROOT="$RUN_ROOT/logs/correct_interventions_4x4"
STATUS="$RUN_ROOT/correct_interventions_parallel.status"
COMPLETE="$RUN_ROOT/correct_interventions.complete"

for path in "$VENV_PYTHON" "$BASE_CONFIG" "$CAUSAL_CONFIG" "$DEFINITION" \
  "$PARALLEL_RUNNER" "$AUDITOR" "$QWEN_SOURCE_RUN" "$GEMMA_SOURCE_RUN" \
  "$ABLATION_SOURCE_RUN"; do
  if [[ ! -e "$path" ]]; then
    echo "Required path is missing: $path" >&2
    exit 2
  fi
done
if [[ -n "$(git -C "$REPO_ROOT" status --porcelain)" ]]; then
  echo "Formal 4+4 run requires a clean repository" >&2
  git -C "$REPO_ROOT" status --short >&2
  exit 2
fi
mkdir -p "$DATASET_ROOT" "$LOG_ROOT"

timestamp() { date -u +'%Y-%m-%dT%H:%M:%SZ'; }

write_status() {
  local state=$1 model=$2 phase=$3 detail=$4
  printf '{"state":"%s","model":"%s","phase":"%s","updated_utc":"%s","detail":"%s"}\n' \
    "$state" "$model" "$phase" "$(timestamp)" "$detail" > "$STATUS"
}

CURRENT_MODEL=campaign
CURRENT_PHASE=initializing
CURRENT_DETAIL="$LOG_ROOT/supervisor.log"
on_error() {
  local rc=$?
  write_status "FAILED:$rc" "$CURRENT_MODEL" "$CURRENT_PHASE" "$CURRENT_DETAIL"
  printf '[%s] FAILED rc=%s model=%s phase=%s detail=%s\n' \
    "$(timestamp)" "$rc" "$CURRENT_MODEL" "$CURRENT_PHASE" "$CURRENT_DETAIL" \
    | tee -a "$LOG_ROOT/supervisor.log" >&2
  exit "$rc"
}
trap on_error ERR

run_python() {
  env \
    HF_HOME="$HF_CACHE" \
    HF_TOKEN_PATH="${HF_TOKEN_PATH:-${HOME}/.cache/huggingface/token}" \
    TOKENIZERS_PARALLELISM=false \
    PYTHONPATH="$REPO_ROOT/src:$REPO_ROOT/scripts" \
    "$VENV_PYTHON" "$@"
}

latest_complete_root() {
  local source_root=$1 model=$2 family=$3 phase=$4
  local parent="$source_root/$model/numeric/causal_v2/$family"
  find "$parent" -maxdepth 2 -type f -path "*/${phase}_*/complete.json" \
    -printf '%T@ %h\n' | sort -n | tail -1 | cut -d' ' -f2-
}

build_model_command() {
  local output_name=$1 model=$2 source=$3 phase=$4 worker_index=${5:-}
  local -n output=$output_name
  local prompt_screen prompt_confirmation answer_screen answer_confirmation
  local ranking_root ablation_confirmation ablation_discovery
  prompt_screen=$(latest_complete_root "$source" "$model" prompt_patching screen)
  prompt_confirmation=$(
    latest_complete_root "$source" "$model" prompt_patching confirmation
  )
  answer_screen=$(latest_complete_root "$source" "$model" answer_patching screen)
  answer_confirmation=$(
    latest_complete_root "$source" "$model" answer_patching confirmation
  )
  ranking_root=$(latest_complete_root "$source" "$model" head_rankings discovery)
  ablation_confirmation=$(
    latest_complete_root \
      "$ABLATION_SOURCE_RUN" "$model" answer_query_head_ablation confirmation
  )
  ablation_discovery=$(
    latest_complete_root \
      "$source" "$model" answer_query_head_ablation screen
  )
  local artifact
  for artifact in \
    "$prompt_screen/selection/prompt_patching_selection.json" \
    "$answer_screen/selection/answer_patching_selection.json" \
    "$prompt_confirmation/detail.csv.gz" \
    "$answer_confirmation/detail.csv.gz" \
    "$ablation_discovery/detail.csv.gz" \
    "$ablation_confirmation/detail.csv.gz" \
    "$ranking_root/head_phenotype_rankings.json"; do
    if [[ ! -s "$artifact" ]]; then
      echo "Resolved source artifact is missing or empty: $artifact" >&2
      return 2
    fi
  done
  output=(
    "$PARALLEL_RUNNER"
    --phase "$phase"
    --worker-count 4
    --run-root "$RUN_ROOT"
    --stimuli "$STIMULI"
    --model "$model"
    --base-config "$BASE_CONFIG"
    --causal-config "$CAUSAL_CONFIG"
    --definition "$DEFINITION"
    --prompt-selection "$prompt_screen/selection/prompt_patching_selection.json"
    --answer-selection "$answer_screen/selection/answer_patching_selection.json"
    --prompt-confirmation-detail "$prompt_confirmation/detail.csv.gz"
    --answer-confirmation-detail "$answer_confirmation/detail.csv.gz"
    --ablation-discovery-detail "$ablation_discovery/detail.csv.gz"
    --ablation-confirmation-detail "$ablation_confirmation/detail.csv.gz"
    --head-rankings "$ranking_root/head_phenotype_rankings.json"
    --cache-dir "$HF_CACHE"
    --device-map auto
    --repo-root "$REPO_ROOT"
    --generation-max-new-tokens 16
  )
  if [[ "$phase" == worker ]]; then
    output+=(--worker-index "$worker_index")
  fi
}

PIDS=()
NAMES=()
start_job() {
  local gpu=$1 name=$2 log_path=$3
  shift 3
  local command=("$@")
  (
    export CUDA_VISIBLE_DEVICES="$gpu"
    printf '[%s] START job=%s gpu=%s\n' "$(timestamp)" "$name" "$gpu" \
      | tee -a "$LOG_ROOT/supervisor.log" "$log_path"
    run_python "${command[@]}" >> "$log_path" 2>&1
    printf '[%s] COMPLETE job=%s gpu=%s\n' "$(timestamp)" "$name" "$gpu" \
      | tee -a "$LOG_ROOT/supervisor.log" "$log_path"
  ) &
  PIDS+=("$!")
  NAMES+=("$name")
}

wait_jobs() {
  local failures=0 index rc
  for index in "${!PIDS[@]}"; do
    rc=0
    wait "${PIDS[$index]}" || rc=$?
    if (( rc != 0 )); then
      printf '[%s] JOB_FAILED job=%s rc=%s\n' \
        "$(timestamp)" "${NAMES[$index]}" "$rc" \
        | tee -a "$LOG_ROOT/supervisor.log" >&2
      failures=$((failures + 1))
    fi
  done
  PIDS=()
  NAMES=()
  if (( failures != 0 )); then
    return 1
  fi
}

cd "$REPO_ROOT"
if [[ ! -s "$STIMULI" ]]; then
  write_status RUNNING campaign freeze "$LOG_ROOT/freeze.log"
  run_python "$REPO_ROOT/scripts/freeze_realistic_niah_v4_4_correct_interventions.py" \
    --output-dir "$DATASET_ROOT" \
    --base-config "$BASE_CONFIG" \
    --definition "$DEFINITION" \
    --cache-dir "$HF_CACHE" > "$LOG_ROOT/freeze.log" 2>&1
else
  write_status RUNNING campaign verify-frozen-stimuli "$LOG_ROOT/freeze.log"
  (cd "$DATASET_ROOT" && sha256sum -c SHA256SUMS_v4_4_causal_v2) \
    >> "$LOG_ROOT/freeze.log" 2>&1
fi

write_status RUNNING campaign gpu-preflight "$LOG_ROOT/gpu_preflight.log"
run_python -c '
import json, torch
minimum = 75 * 1024**3
count = torch.cuda.device_count()
devices = [
    {
        "index": i,
        "name": torch.cuda.get_device_name(i),
        "total_memory_bytes": int(torch.cuda.get_device_properties(i).total_memory),
    }
    for i in range(count)
]
print(json.dumps({"cuda_device_count": count, "devices": devices}, indent=2))
if count < 8:
    raise SystemExit(f"4+4 launcher requires at least 8 visible GPUs; found {count}")
undersized = [row for row in devices[:8] if row["total_memory_bytes"] < minimum]
if undersized:
    raise SystemExit(f"The first eight GPUs must each expose at least 75 GiB: {undersized}")
' > "$LOG_ROOT/gpu_preflight.log" 2>&1

# The preparation gate is model-dependent but cheap. Running the two models in
# parallel freezes one unique baseline-only seed prefix and work plan per model.
write_status RUNNING campaign prepare "Qwen GPU0; Gemma GPU4"
CURRENT_PHASE=prepare
build_model_command qwen_prepare Qwen3-8B "$QWEN_SOURCE_RUN" prepare
build_model_command gemma_prepare Gemma4-E4B "$GEMMA_SOURCE_RUN" prepare
start_job 0 Qwen3-8B.prepare "$LOG_ROOT/Qwen3-8B.prepare.log" \
  "${qwen_prepare[@]}"
start_job 4 Gemma4-E4B.prepare "$LOG_ROOT/Gemma4-E4B.prepare.log" \
  "${gemma_prepare[@]}"
wait_jobs

# Eight model replicas now consume disjoint frozen work assignments. No worker
# writes into another worker directory or the canonical merged result paths.
write_status RUNNING campaign workers "Qwen GPUs0-3; Gemma GPUs4-7"
CURRENT_PHASE=workers
for worker in 0 1 2 3; do
  build_model_command \
    qwen_worker Qwen3-8B "$QWEN_SOURCE_RUN" worker "$worker"
  build_model_command \
    gemma_worker Gemma4-E4B "$GEMMA_SOURCE_RUN" worker "$worker"
  start_job "$worker" "Qwen3-8B.worker${worker}" \
    "$LOG_ROOT/Qwen3-8B.worker${worker}.log" "${qwen_worker[@]}"
  start_job "$((worker + 4))" "Gemma4-E4B.worker${worker}" \
    "$LOG_ROOT/Gemma4-E4B.worker${worker}.log" "${gemma_worker[@]}"
done
wait_jobs

# Merge is CPU-only and refuses missing, overlapping, or modified worker data.
write_status RUNNING campaign merge "$LOG_ROOT/merge.log"
CURRENT_PHASE=merge
build_model_command qwen_merge Qwen3-8B "$QWEN_SOURCE_RUN" merge
build_model_command gemma_merge Gemma4-E4B "$GEMMA_SOURCE_RUN" merge
start_job '' Qwen3-8B.merge "$LOG_ROOT/Qwen3-8B.merge.log" \
  "${qwen_merge[@]}"
start_job '' Gemma4-E4B.merge "$LOG_ROOT/Gemma4-E4B.merge.log" \
  "${gemma_merge[@]}"
wait_jobs

write_status RUNNING campaign audit "$LOG_ROOT/audit.log"
CURRENT_PHASE=audit
CUDA_VISIBLE_DEVICES='' run_python "$AUDITOR" \
  --run-root "$RUN_ROOT" \
  --definition "$DEFINITION" > "$LOG_ROOT/audit.log" 2>&1

printf '{"status":"complete","layout":"4+4","completed_utc":"%s","audit":"%s"}\n' \
  "$(timestamp)" "$RUN_ROOT/audit/correct_interventions/audit.json" > "$COMPLETE"
write_status COMPLETE campaign all "$LOG_ROOT/supervisor.log"
printf '[%s] CAMPAIGN COMPLETE layout=4+4\n' "$(timestamp)" \
  | tee -a "$LOG_ROOT/supervisor.log"
