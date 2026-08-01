#!/usr/bin/env bash
set -Eeuo pipefail

usage() {
  cat <<'EOF'
Usage:
  launch_realistic_niah_v4_causal_campaign.sh \
    RUN_ROOT VENV_PYTHON HF_CACHE [MODEL ...]

The script is intended to be detached with setsid/nohup. It runs the targeted
V4 causal screen_8h_v1 profile sequentially on one GPU. The screen retains all
four panels and all ten confirmation seeds while narrowing counts, intervention
families, and layers. Existing shards are validated and reused by the Python
pipeline, so an interrupted campaign can be restarted without deleting prior
captures.
EOF
}

if [[ $# -lt 3 ]]; then
  usage >&2
  exit 2
fi

RUN_ROOT=$1
VENV_PYTHON=$2
HF_CACHE=$3
shift 3

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd -- "$SCRIPT_DIR/.." && pwd)
STIMULI="$RUN_ROOT/dataset/stimuli.jsonl"
CONFIG="$REPO_ROOT/configs/realistic_niah_v4.json"
LOG_ROOT="$RUN_ROOT/logs"
CAMPAIGN_STATUS="$RUN_ROOT/causal_campaign.status"
PROFILE=screen_8h_v1
COMPLETE_STATUS="$RUN_ROOT/causal_screen_8h.complete"
SCREEN_VARIANTS=v4.1,v4.2,v4.3,v4.4
SCREEN_SEEDS=1254,1255,1256,1257,1258,1259,1260,1261,1262,1263
if [[ $# -gt 0 ]]; then
  MODELS=("$@")
else
  MODELS=(Qwen3-8B Gemma4-E4B)
fi
STAGES=(ablation patching geometric-steering)

if [[ ! -x "$VENV_PYTHON" ]]; then
  echo "Venv Python is not executable: $VENV_PYTHON" >&2
  exit 2
fi
if [[ ! -s "$STIMULI" ]]; then
  echo "Stimulus file is missing or empty: $STIMULI" >&2
  exit 2
fi
if [[ ! -s "$CONFIG" ]]; then
  echo "Config is missing or empty: $CONFIG" >&2
  exit 2
fi
mkdir -p "$LOG_ROOT"

timestamp() { date -u +'%Y-%m-%dT%H:%M:%SZ'; }

write_status() {
  local path=$1 state=$2 model=$3 stage=$4 log_path=$5
  printf '{"state":"%s","profile":"%s","model":"%s","stage":"%s","updated_utc":"%s","log":"%s"}\n' \
    "$state" "$PROFILE" "$model" "$stage" "$(timestamp)" "$log_path" > "$path"
}

screen_layers() {
  case "$1" in
    Qwen3-8B) printf '9,18,26' ;;
    Gemma4-E4B) printf '10,20,31' ;;
    *) echo "No screen_8h_v1 layer map for model: $1" >&2; return 2 ;;
  esac
}

run_stage() {
  local model=$1 stage=$2
  local safe_stage=${stage//-/_}
  local status_path="$RUN_ROOT/causal_${PROFILE}_${safe_stage}_${model}.status"
  local log_path="$LOG_ROOT/causal_${PROFILE}_${safe_stage}_${model}.log"
  local -a extra=()
  case "$stage" in
    preflight)
      extra+=(--variants v4.1 --seeds 1254 --counts 1 --forward-smoke)
      ;;
    ablation)
      extra+=(
        --variants "$SCREEN_VARIANTS" --seeds "$SCREEN_SEEDS"
        --counts 7,8,9,10
        --ablation-poolings span_end
        --ablation-scopes answer_query
        --causal-top-ns 4,8
        --causal-random-replicates 1
      )
      ;;
    patching)
      extra+=(
        --variants "$SCREEN_VARIANTS" --seeds "$SCREEN_SEEDS"
        --causal-layers "$(screen_layers "$model")"
        --causal-count-pairs 5:6,7:8,9:10
        --residual-patch-sites toggled_needle_end
        --residual-patch-protocols cumulative_from_layer
      )
      ;;
    geometric-steering)
      extra+=(
        --variants "$SCREEN_VARIANTS" --seeds "$SCREEN_SEEDS"
        --causal-layers "$(screen_layers "$model")"
        --steering-count-pairs 7:8,9:10,5:10
        --steering-methods centroid_delta
        --steering-alphas 1
        --steering-random-replicates 1
      )
      ;;
    *)
      echo "Unknown campaign stage: $stage" >&2
      return 2
      ;;
  esac

  write_status "$status_path" RUNNING "$model" "$stage" "$log_path"
  printf '[%s] START model=%s stage=%s\n' "$(timestamp)" "$model" "$stage" | tee -a "$log_path"
  if env \
      HF_HOME="$HF_CACHE" \
      PYTHONPATH="$REPO_ROOT/src" \
      "$VENV_PYTHON" "$REPO_ROOT/scripts/run_realistic_niah_v4.py" \
        --stage "$stage" \
        --stimuli "$STIMULI" \
        --config "$CONFIG" \
        --output-dir "$RUN_ROOT" \
        --model "$model" \
        --answer-format numeric \
        --cache-dir "$HF_CACHE" \
        --device-map auto \
        --repo-root "$REPO_ROOT" \
        "${extra[@]}" >> "$log_path" 2>&1; then
    write_status "$status_path" COMPLETE "$model" "$stage" "$log_path"
    printf '[%s] COMPLETE model=%s stage=%s\n' "$(timestamp)" "$model" "$stage" | tee -a "$log_path"
  else
    local rc=$?
    write_status "$status_path" "FAILED:$rc" "$model" "$stage" "$log_path"
    write_status "$CAMPAIGN_STATUS" "FAILED:$rc" "$model" "$stage" "$log_path"
    printf '[%s] FAILED rc=%s model=%s stage=%s\n' "$(timestamp)" "$rc" "$model" "$stage" | tee -a "$log_path" >&2
    return "$rc"
  fi
}

write_status "$CAMPAIGN_STATUS" RUNNING campaign preflight "$LOG_ROOT"
printf '[%s] PROFILE=%s expected_generations_per_model=2800\n' \
  "$(timestamp)" "$PROFILE"

# Validate every retained intervention hook on both real checkpoints. Stages
# are outermost so cross-model screen comparisons become available early.
for model in "${MODELS[@]}"; do
  run_stage "$model" preflight
done
for stage in "${STAGES[@]}"; do
  for model in "${MODELS[@]}"; do
    write_status "$CAMPAIGN_STATUS" RUNNING "$model" "$stage" "$LOG_ROOT"
    run_stage "$model" "$stage"
  done
done

write_status "$CAMPAIGN_STATUS" COMPLETE campaign all "$LOG_ROOT"
write_status "$COMPLETE_STATUS" COMPLETE campaign all "$LOG_ROOT"
printf '[%s] COMPLETE causal campaign profile=%s\n' "$(timestamp)" "$PROFILE"
