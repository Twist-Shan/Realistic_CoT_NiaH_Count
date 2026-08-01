#!/usr/bin/env bash
set -Eeuo pipefail

usage() {
  cat <<'EOF'
Usage:
  launch_realistic_niah_v4_causal_campaign.sh \
    RUN_ROOT VENV_PYTHON HF_CACHE [MODEL ...]

The script is intended to be detached with setsid/nohup. It runs the registered
V4 causal stages sequentially on one GPU. Existing shards are validated and
reused by the Python pipeline, so an interrupted campaign can be restarted
without deleting prior captures.
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
if [[ $# -gt 0 ]]; then
  MODELS=("$@")
else
  MODELS=(Qwen3-8B Gemma4-E4B)
fi
STAGES=(ablation head-patching patching geometric-steering)

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
  printf '{"state":"%s","model":"%s","stage":"%s","updated_utc":"%s","log":"%s"}\n' \
    "$state" "$model" "$stage" "$(timestamp)" "$log_path" > "$path"
}

run_stage() {
  local model=$1 stage=$2
  local safe_stage=${stage//-/_}
  local status_path="$RUN_ROOT/causal_${safe_stage}_${model}.status"
  local log_path="$LOG_ROOT/causal_${safe_stage}_${model}.log"
  local -a extra=()
  if [[ "$stage" == preflight ]]; then
    extra+=(--variants v4.1 --seeds 1254 --counts 1 --forward-smoke)
  fi

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

# Validate every intervention hook on both real checkpoints before the formal
# sweeps. Stages are outermost so cross-model comparisons become available as
# early as possible on a single GPU.
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
printf '[%s] COMPLETE causal campaign\n' "$(timestamp)"
