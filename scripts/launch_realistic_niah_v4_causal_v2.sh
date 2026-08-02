#!/usr/bin/env bash
set -Eeuo pipefail

usage() {
  cat <<'EOF'
Usage:
  launch_realistic_niah_v4_causal_v2.sh \
    RUN_ROOT VENV_PYTHON HF_CACHE BASE_STIMULI ATTENTION_SOURCE [MODEL ...]

Runs the frozen V4.4 causal-v2 campaign.  The formal sequence is:
N=0 baseline extension; answer-query top-1..32 head ablation; prompt and
answer residual-patching screens; all-layer centroid fit and steering screen;
five-seed stability selection; exact five-seed held-out confirmation for every
selected layer/k; strict final audit.  Existing hash-matched shards are reused.
EOF
}

if [[ $# -lt 5 ]]; then
  usage >&2
  exit 2
fi

RUN_ROOT=$1
VENV_PYTHON=$2
HF_CACHE=$3
BASE_STIMULI=$4
ATTENTION_SOURCE=$5
shift 5

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd -- "$SCRIPT_DIR/.." && pwd)
BASE_CONFIG="$REPO_ROOT/configs/realistic_niah_v4.json"
CAUSAL_CONFIG="$REPO_ROOT/configs/realistic_niah_v4_causal_v2.json"
DATASET_ROOT="$RUN_ROOT/dataset/causal_v2"
STIMULI="$DATASET_ROOT/stimuli_v4_4_causal_v2.jsonl"
LOG_ROOT="$RUN_ROOT/logs/causal_v2"
STATUS="$RUN_ROOT/causal_v2.status"
COMPLETE="$RUN_ROOT/causal_v2.complete"

if [[ $# -gt 0 ]]; then
  MODELS=("$@")
else
  MODELS=(Qwen3-8B Gemma4-E4B)
fi

if [[ ! -x "$VENV_PYTHON" ]]; then
  echo "Venv Python is not executable: $VENV_PYTHON" >&2
  exit 2
fi
if [[ ! -s "$BASE_STIMULI" ]]; then
  echo "Base V4 stimulus file is missing or empty: $BASE_STIMULI" >&2
  exit 2
fi
if [[ "$ATTENTION_SOURCE" != AUTO && ! -e "$ATTENTION_SOURCE" ]]; then
  echo "Attention source does not exist: $ATTENTION_SOURCE" >&2
  exit 2
fi
mkdir -p "$DATASET_ROOT" "$LOG_ROOT"

timestamp() { date -u +'%Y-%m-%dT%H:%M:%SZ'; }

write_status() {
  local state=$1 model=$2 stage=$3 log_path=$4
  printf '{"state":"%s","model":"%s","stage":"%s","updated_utc":"%s","log":"%s"}\n' \
    "$state" "$model" "$stage" "$(timestamp)" "$log_path" > "$STATUS"
}

CURRENT_MODEL=campaign
CURRENT_STAGE=initializing
CURRENT_LOG="$LOG_ROOT/campaign.log"
on_error() {
  local rc=$?
  write_status "FAILED:$rc" "$CURRENT_MODEL" "$CURRENT_STAGE" "$CURRENT_LOG"
  printf '[%s] FAILED rc=%s model=%s stage=%s\n' \
    "$(timestamp)" "$rc" "$CURRENT_MODEL" "$CURRENT_STAGE" | tee -a "$CURRENT_LOG" >&2
  exit "$rc"
}
trap on_error ERR

run_python() {
  env \
    HF_HOME="$HF_CACHE" \
    HF_TOKEN_PATH="${HF_TOKEN_PATH:-${HOME}/.cache/huggingface/token}" \
    TOKENIZERS_PARALLELISM=false \
    PYTHONPATH="$REPO_ROOT/src" \
    "$VENV_PYTHON" "$@"
}

run_logged() {
  local model=$1 stage=$2 log_path=$3
  shift 3
  CURRENT_MODEL=$model
  CURRENT_STAGE=$stage
  CURRENT_LOG=$log_path
  write_status RUNNING "$model" "$stage" "$log_path"
  printf '[%s] START model=%s stage=%s\n' "$(timestamp)" "$model" "$stage" | tee -a "$log_path"
  run_python "$@" >> "$log_path" 2>&1
  write_status COMPLETE "$model" "$stage" "$log_path"
  printf '[%s] COMPLETE model=%s stage=%s\n' "$(timestamp)" "$model" "$stage" | tee -a "$log_path"
}

latest_complete_root() {
  local model=$1 family=$2 phase=$3
  local parent="$RUN_ROOT/$model/numeric/causal_v2/$family"
  find "$parent" -maxdepth 2 -type f -path "*/${phase}_*/complete.json" \
    -printf '%T@ %h\n' | sort -n | tail -1 | cut -d' ' -f2-
}

selected_count() {
  local selection=$1
  "$VENV_PYTHON" -c \
    'import json,sys; print(int(json.load(open(sys.argv[1], encoding="utf-8"))["selected_condition_count"]))' \
    "$selection"
}

common_args() {
  local model=$1
  printf '%s\n' \
    --run-root "$RUN_ROOT" \
    --stimuli "$STIMULI" \
    --model "$model" \
    --config "$BASE_CONFIG" \
    --causal-config "$CAUSAL_CONFIG" \
    --cache-dir "$HF_CACHE" \
    --device-map auto \
    --repo-root "$REPO_ROOT" \
    --generation-max-new-tokens 16
}

if [[ ! -s "$STIMULI" ]]; then
  run_logged campaign freeze "$LOG_ROOT/freeze.log" \
    "$REPO_ROOT/scripts/freeze_realistic_niah_v4_causal_v2.py" \
      --output-dir "$DATASET_ROOT" \
      --config "$BASE_CONFIG" \
      --base-stimuli "$BASE_STIMULI" \
      --cache-dir "$HF_CACHE"
else
  CURRENT_MODEL=campaign
  CURRENT_STAGE=verify-frozen-stimuli
  CURRENT_LOG="$LOG_ROOT/freeze.log"
  (cd "$DATASET_ROOT" && sha256sum -c SHA256SUMS_v4_4_causal_v2) \
    >> "$CURRENT_LOG" 2>&1
fi

for model in "${MODELS[@]}"; do
  mapfile -t COMMON < <(common_args "$model")
  MODEL_LOG_ROOT="$LOG_ROOT/$model"
  mkdir -p "$MODEL_LOG_ROOT"

  run_logged "$model" prompt-alignment "$MODEL_LOG_ROOT/prompt_alignment.log" \
    "$REPO_ROOT/scripts/run_realistic_niah_v4_causal_v2.py" \
      --stage prompt-alignment "${COMMON[@]}"
done

for model in "${MODELS[@]}"; do
  mapfile -t COMMON < <(common_args "$model")
  MODEL_LOG_ROOT="$LOG_ROOT/$model"
  mkdir -p "$MODEL_LOG_ROOT"

  run_logged "$model" baseline "$MODEL_LOG_ROOT/baseline.log" \
    "$REPO_ROOT/scripts/run_realistic_niah_v4_causal_v2.py" \
      --stage baseline "${COMMON[@]}"
  BASELINE_ROOT=$(latest_complete_root "$model" baseline all)
  BASELINE="$BASELINE_ROOT/generation_labels.csv"

  run_logged "$model" head-rankings "$MODEL_LOG_ROOT/head_rankings.log" \
    "$REPO_ROOT/scripts/run_realistic_niah_v4_causal_v2.py" \
      --stage head-rankings "${COMMON[@]}" \
      --attention-source "$ATTENTION_SOURCE"
  RANKING_ROOT=$(latest_complete_root "$model" head_rankings discovery)
  RANKINGS="$RANKING_ROOT/head_phenotype_rankings.json"

  run_logged "$model" ablation "$MODEL_LOG_ROOT/ablation.log" \
    "$REPO_ROOT/scripts/run_realistic_niah_v4_causal_v2.py" \
      --stage ablation "${COMMON[@]}" \
      --baseline-labels "$BASELINE" \
      --head-rankings "$RANKINGS"
  ABLATION_ROOT=$(latest_complete_root "$model" answer_query_head_ablation screen)
  ABLATION_DETAIL="$ABLATION_ROOT/detail.csv.gz"
  PROMPT_CONFIRMATION_STATS=
  ANSWER_CONFIRMATION_STATS=
  STEERING_CONFIRMATION_STATS=

  for patch_stage in prompt-patching answer-patching; do
    if [[ "$patch_stage" == prompt-patching ]]; then
      family=prompt_patching
    else
      family=answer_patching
    fi
    run_logged "$model" "${family}-screen" "$MODEL_LOG_ROOT/${family}_screen.log" \
      "$REPO_ROOT/scripts/run_realistic_niah_v4_causal_v2.py" \
        --stage "$patch_stage" --phase screen "${COMMON[@]}" \
        --baseline-labels "$BASELINE"
    SCREEN_ROOT=$(latest_complete_root "$model" "$family" screen)
    SCREEN_DETAIL="$SCREEN_ROOT/detail.csv.gz"
    if [[ "$family" == prompt_patching ]]; then
      PROMPT_SCREEN_DETAIL=$SCREEN_DETAIL
    else
      ANSWER_SCREEN_DETAIL=$SCREEN_DETAIL
    fi
    run_logged "$model" "${family}-select" "$MODEL_LOG_ROOT/${family}_select.log" \
      "$REPO_ROOT/scripts/run_realistic_niah_v4_causal_v2.py" \
        --stage select --family "$family" "${COMMON[@]}" \
        --detail "$SCREEN_DETAIL"
    SELECTION="$SCREEN_ROOT/selection/${family}_selection.json"
    if [[ $(selected_count "$SELECTION") -gt 0 ]]; then
      run_logged "$model" "${family}-confirmation" "$MODEL_LOG_ROOT/${family}_confirmation.log" \
        "$REPO_ROOT/scripts/run_realistic_niah_v4_causal_v2.py" \
          --stage "$patch_stage" --phase confirmation "${COMMON[@]}" \
          --baseline-labels "$BASELINE" \
          --selection-json "$SELECTION"
      CONFIRMATION_ROOT=$(latest_complete_root "$model" "$family" confirmation)
      run_logged "$model" "${family}-stats" "$MODEL_LOG_ROOT/${family}_stats.log" \
        "$REPO_ROOT/scripts/run_realistic_niah_v4_causal_v2.py" \
          --stage confirmation-stats --family "$family" "${COMMON[@]}" \
          --screen-detail "$SCREEN_DETAIL" \
          --confirmation-detail "$CONFIRMATION_ROOT/detail.csv.gz" \
          --selection-json "$SELECTION"
      if [[ "$family" == prompt_patching ]]; then
        PROMPT_CONFIRMATION_STATS="$CONFIRMATION_ROOT/analysis/${family}_confirmation_statistics.csv"
      else
        ANSWER_CONFIRMATION_STATS="$CONFIRMATION_ROOT/analysis/${family}_confirmation_statistics.csv"
      fi
    else
      printf '{"family":"%s","model":"%s","reason":"no condition passed the frozen screen"}\n' \
        "$family" "$model" > "$SCREEN_ROOT/selection/no_confirmation_required.json"
    fi
  done

  run_logged "$model" steering-centroids "$MODEL_LOG_ROOT/steering_centroids.log" \
    "$REPO_ROOT/scripts/run_realistic_niah_v4_causal_v2.py" \
      --stage steering-centroids "${COMMON[@]}" \
      --baseline-labels "$BASELINE"
  CENTROID_ROOT=$(latest_complete_root "$model" steering_centroids fit)
  CENTROIDS="$CENTROID_ROOT/centroids.npz"

  run_logged "$model" steering-screen "$MODEL_LOG_ROOT/steering_screen.log" \
    "$REPO_ROOT/scripts/run_realistic_niah_v4_causal_v2.py" \
      --stage steering --phase screen "${COMMON[@]}" \
      --baseline-labels "$BASELINE" \
      --centroids "$CENTROIDS"
  STEERING_SCREEN_ROOT=$(latest_complete_root "$model" steering screen)
  STEERING_SCREEN_DETAIL="$STEERING_SCREEN_ROOT/detail.csv.gz"
  run_logged "$model" steering-select "$MODEL_LOG_ROOT/steering_select.log" \
    "$REPO_ROOT/scripts/run_realistic_niah_v4_causal_v2.py" \
      --stage select --family steering "${COMMON[@]}" \
      --detail "$STEERING_SCREEN_DETAIL"
  STEERING_SELECTION="$STEERING_SCREEN_ROOT/selection/steering_selection.json"
  if [[ $(selected_count "$STEERING_SELECTION") -gt 0 ]]; then
    run_logged "$model" steering-confirmation "$MODEL_LOG_ROOT/steering_confirmation.log" \
      "$REPO_ROOT/scripts/run_realistic_niah_v4_causal_v2.py" \
        --stage steering --phase confirmation "${COMMON[@]}" \
        --baseline-labels "$BASELINE" \
        --centroids "$CENTROIDS" \
        --selection-json "$STEERING_SELECTION"
    STEERING_CONFIRMATION_ROOT=$(latest_complete_root "$model" steering confirmation)
    run_logged "$model" steering-stats "$MODEL_LOG_ROOT/steering_stats.log" \
      "$REPO_ROOT/scripts/run_realistic_niah_v4_causal_v2.py" \
        --stage confirmation-stats --family steering "${COMMON[@]}" \
        --screen-detail "$STEERING_SCREEN_DETAIL" \
        --confirmation-detail "$STEERING_CONFIRMATION_ROOT/detail.csv.gz" \
        --selection-json "$STEERING_SELECTION"
    STEERING_CONFIRMATION_STATS="$STEERING_CONFIRMATION_ROOT/analysis/steering_confirmation_statistics.csv"
  else
    printf '{"family":"steering","model":"%s","reason":"no condition passed the frozen screen"}\n' \
      "$model" > "$STEERING_SCREEN_ROOT/selection/no_confirmation_required.json"
  fi

  ANALYSIS_ARGS=(
    --ablation-detail "$ABLATION_DETAIL"
    --prompt-screen-detail "$PROMPT_SCREEN_DETAIL"
    --answer-screen-detail "$ANSWER_SCREEN_DETAIL"
    --steering-screen-detail "$STEERING_SCREEN_DETAIL"
    --output-dir "$RUN_ROOT/$model/numeric/causal_v2/analysis"
    --bootstrap-repetitions 10000
  )
  if [[ -n "$PROMPT_CONFIRMATION_STATS" ]]; then
    ANALYSIS_ARGS+=(--prompt-confirmation-statistics "$PROMPT_CONFIRMATION_STATS")
  fi
  if [[ -n "$ANSWER_CONFIRMATION_STATS" ]]; then
    ANALYSIS_ARGS+=(--answer-confirmation-statistics "$ANSWER_CONFIRMATION_STATS")
  fi
  if [[ -n "$STEERING_CONFIRMATION_STATS" ]]; then
    ANALYSIS_ARGS+=(--steering-confirmation-statistics "$STEERING_CONFIRMATION_STATS")
  fi
  run_logged "$model" analysis "$MODEL_LOG_ROOT/analysis.log" \
    "$REPO_ROOT/scripts/analyze_realistic_niah_v4_causal_v2.py" \
      "${ANALYSIS_ARGS[@]}"

  run_logged "$model" audit "$MODEL_LOG_ROOT/audit.log" \
    "$REPO_ROOT/scripts/audit_realistic_niah_v4_causal_v2.py" \
      --run-root "$RUN_ROOT" \
      --model "$model" \
      --stimuli "$STIMULI" \
      --causal-config "$CAUSAL_CONFIG" \
      --require-confirmation
done

printf '{"state":"COMPLETE","models":"%s","updated_utc":"%s"}\n' \
  "${MODELS[*]}" "$(timestamp)" > "$STATUS"
printf '{"state":"COMPLETE","models":"%s","updated_utc":"%s"}\n' \
  "${MODELS[*]}" "$(timestamp)" > "$COMPLETE"
printf '[%s] COMPLETE causal-v2 campaign models=%s\n' "$(timestamp)" "${MODELS[*]}" | tee -a "$LOG_ROOT/campaign.log"
