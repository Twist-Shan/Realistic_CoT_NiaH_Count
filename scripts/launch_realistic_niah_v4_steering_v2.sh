#!/usr/bin/env bash
set -Eeuo pipefail

usage() {
  cat <<'EOF'
Usage:
  launch_realistic_niah_v4_steering_v2.sh RUN_ROOT VENV_PYTHON HF_CACHE [MODEL ...]

Runs a restartable discovery -> locked-confirmation steering follow-up.
Discovery uses seeds 1234-1237 to select one single-layer and one multi-layer
centroid-delta plan per model with a worst-panel robustness objective.
Confirmation uses all ten held-out seeds 1254-1263. Existing family shards are
validated and reused on restart.
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
CAMPAIGN_STATUS="$RUN_ROOT/steering_v2.status"
COMPLETE_STATUS="$RUN_ROOT/steering_v2.complete"
SCREEN_SEEDS=1234,1235,1236,1237
CONFIRMATION_SEEDS=1254,1255,1256,1257,1258,1259,1260,1261,1262,1263
VARIANTS=v4.1,v4.2,v4.3,v4.4
PAIRS=7:8,9:10,5:10
ALPHAS=0.25,0.5,1
if [[ $# -gt 0 ]]; then
  MODELS=("$@")
else
  MODELS=(Qwen3-8B Gemma4-E4B)
fi

if [[ ! -x "$VENV_PYTHON" ]]; then
  echo "Venv Python is not executable: $VENV_PYTHON" >&2
  exit 2
fi
if [[ ! -s "$STIMULI" ]]; then
  echo "Stimulus file is missing or empty: $STIMULI" >&2
  exit 2
fi
mkdir -p "$LOG_ROOT"

timestamp() { date -u +'%Y-%m-%dT%H:%M:%SZ'; }

write_status() {
  local state=$1 model=$2 phase=$3 log_path=$4
  printf '{"state":"%s","model":"%s","phase":"%s","updated_utc":"%s","log":"%s"}\n' \
    "$state" "$model" "$phase" "$(timestamp)" "$log_path" > "$CAMPAIGN_STATUS"
}

centroids_for() {
  case "$1" in
    Qwen3-8B)
      printf '%s' "$RUN_ROOT/Qwen3-8B/numeric/causal/geometric_steering_v1/design_cf77fd4452c2/centroids.npz"
      ;;
    Gemma4-E4B)
      printf '%s' "$RUN_ROOT/Gemma4-E4B/numeric/causal/geometric_steering_v1/design_28163399d9ee/centroids.npz"
      ;;
    *)
      echo "No registered centroid bundle for model: $1" >&2
      return 2
      ;;
  esac
}

layer_sets_for() {
  case "$1" in
    Qwen3-8B) printf '9,18,26,18+26,9+18+26' ;;
    Gemma4-E4B) printf '10,20,31,20+31,10+20+31' ;;
    *) echo "No registered layer-set screen for model: $1" >&2; return 2 ;;
  esac
}

run_phase() {
  local model=$1 phase=$2 selection=${3:-}
  local log_path="$LOG_ROOT/steering_v2_${model}_${phase}.log"
  local centroids
  centroids=$(centroids_for "$model")
  local -a phase_args
  if [[ "$phase" == screen ]]; then
    phase_args=(
      --seeds "$SCREEN_SEEDS"
      --layer-sets "$(layer_sets_for "$model")"
      --alphas "$ALPHAS"
    )
  else
    phase_args=(
      --seeds "$CONFIRMATION_SEEDS"
      --selection-json "$selection"
    )
  fi
  write_status RUNNING "$model" "$phase" "$log_path"
  printf '[%s] START model=%s phase=%s\n' "$(timestamp)" "$model" "$phase" | tee -a "$log_path"
  if env \
      HF_HOME="$HF_CACHE" \
      HF_TOKEN_PATH="${HF_TOKEN_PATH:-${HOME}/.cache/huggingface/token}" \
      TOKENIZERS_PARALLELISM=false \
      PYTHONPATH="$REPO_ROOT/src" \
      "$VENV_PYTHON" "$REPO_ROOT/scripts/run_realistic_niah_v4_steering_v2.py" \
        --phase "$phase" \
        --stimuli "$STIMULI" \
        --config "$CONFIG" \
        --run-root "$RUN_ROOT" \
        --model "$model" \
        --centroids "$centroids" \
        --variants "$VARIANTS" \
        --count-pairs "$PAIRS" \
        --random-replicates 1 \
        --generation-max-new-tokens 16 \
        --cache-dir "$HF_CACHE" \
        --device-map auto \
        --repo-root "$REPO_ROOT" \
        "${phase_args[@]}" >> "$log_path" 2>&1; then
    write_status COMPLETE "$model" "$phase" "$log_path"
    printf '[%s] COMPLETE model=%s phase=%s\n' "$(timestamp)" "$model" "$phase" | tee -a "$log_path"
  else
    local rc=$?
    write_status "FAILED:$rc" "$model" "$phase" "$log_path"
    printf '[%s] FAILED rc=%s model=%s phase=%s\n' \
      "$(timestamp)" "$rc" "$model" "$phase" | tee -a "$log_path" >&2
    return "$rc"
  fi
}

latest_selection() {
  local model=$1
  find "$RUN_ROOT/$model/numeric/causal/geometric_steering_v2" \
    -type f -path '*/screen_*/selection.json' -printf '%T@ %p\n' \
    | sort -n | tail -1 | cut -d' ' -f2-
}

printf '[%s] steering_v2 expected_rows_per_model=3840\n' "$(timestamp)"
for model in "${MODELS[@]}"; do
  run_phase "$model" screen
  selection=$(latest_selection "$model")
  if [[ ! -s "$selection" ]]; then
    echo "No selection JSON found after $model discovery screen" >&2
    exit 1
  fi
  run_phase "$model" confirmation "$selection"
done

printf '{"state":"COMPLETE","models":"%s","updated_utc":"%s"}\n' \
  "${MODELS[*]}" "$(timestamp)" > "$CAMPAIGN_STATUS"
printf '{"state":"COMPLETE","models":"%s","updated_utc":"%s"}\n' \
  "${MODELS[*]}" "$(timestamp)" > "$COMPLETE_STATUS"
printf '[%s] COMPLETE steering_v2 campaign\n' "$(timestamp)"
