#!/usr/bin/env bash
set -Eeuo pipefail

usage() {
  cat <<'EOF'
Usage:
  launch_realistic_niah_v4_answer_query_patching.sh \
    RUN_ROOT VENV_PYTHON HF_CACHE [MODEL ...]

Run the registered answer-query residual-patching screen on the V4 numeric
confirmation split. Existing design shards are validated and reused, so the
campaign can be restarted without deleting prior outputs.
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
PROFILE=answer_query_dense_v1
CAMPAIGN_STATUS="$RUN_ROOT/answer_query_patching.status"
COMPLETE_STATUS="$RUN_ROOT/answer_query_patching_dense_v1.complete"
VARIANTS=v4.1,v4.2,v4.3,v4.4
SEEDS=1254,1255,1256,1257,1258,1259,1260,1261,1262,1263
COUNT_PAIRS=5:6,7:8,9:10,5:10

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
if [[ ! -s "$CONFIG" ]]; then
  echo "Config is missing or empty: $CONFIG" >&2
  exit 2
fi
mkdir -p "$LOG_ROOT"

timestamp() { date -u +'%Y-%m-%dT%H:%M:%SZ'; }

write_status() {
  local state=$1 model=$2 log_path=$3
  printf '{"state":"%s","profile":"%s","model":"%s","stage":"patching","updated_utc":"%s","log":"%s"}\n' \
    "$state" "$PROFILE" "$model" "$(timestamp)" "$log_path" > "$CAMPAIGN_STATUS"
}

screen_layers() {
  case "$1" in
    Qwen3-8B) printf '0,9,18,26,29,32,34,35' ;;
    Gemma4-E4B) printf '0,10,20,31,35,38,40,41' ;;
    *) echo "No $PROFILE layer map for model: $1" >&2; return 2 ;;
  esac
}

for model in "${MODELS[@]}"; do
  log_path="$LOG_ROOT/${PROFILE}_${model}.log"
  write_status RUNNING "$model" "$log_path"
  printf '[%s] START profile=%s model=%s expected_families=40 expected_rows=2560\n' \
    "$(timestamp)" "$PROFILE" "$model" | tee -a "$log_path"
  if env \
      HF_HOME="$HF_CACHE" \
      PYTHONPATH="$REPO_ROOT/src" \
      "$VENV_PYTHON" "$REPO_ROOT/scripts/run_realistic_niah_v4.py" \
        --stage patching \
        --stimuli "$STIMULI" \
        --config "$CONFIG" \
        --output-dir "$RUN_ROOT" \
        --model "$model" \
        --answer-format numeric \
        --cache-dir "$HF_CACHE" \
        --device-map auto \
        --repo-root "$REPO_ROOT" \
        --variants "$VARIANTS" \
        --seeds "$SEEDS" \
        --causal-layers "$(screen_layers "$model")" \
        --causal-count-pairs "$COUNT_PAIRS" \
        --residual-patch-sites answer_query \
        --residual-patch-protocols single_layer >> "$log_path" 2>&1; then
    printf '[%s] COMPLETE profile=%s model=%s\n' \
      "$(timestamp)" "$PROFILE" "$model" | tee -a "$log_path"
  else
    rc=$?
    write_status "FAILED:$rc" "$model" "$log_path"
    printf '[%s] FAILED rc=%s profile=%s model=%s\n' \
      "$(timestamp)" "$rc" "$PROFILE" "$model" | tee -a "$log_path" >&2
    exit "$rc"
  fi
done

write_status COMPLETE campaign "$LOG_ROOT"
printf '{"state":"COMPLETE","profile":"%s","model":"campaign","stage":"patching","updated_utc":"%s","log":"%s"}\n' \
  "$PROFILE" "$(timestamp)" "$LOG_ROOT" > "$COMPLETE_STATUS"
printf '[%s] COMPLETE answer-query patching profile=%s\n' "$(timestamp)" "$PROFILE"
