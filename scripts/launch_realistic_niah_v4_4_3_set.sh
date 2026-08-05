#!/usr/bin/env bash
set -Eeuo pipefail

if [[ $# -lt 1 ]]; then
  echo "Usage: launch_realistic_niah_v4_4_3_set.sh RUN_ID [--resume] [MODEL]" >&2
  exit 2
fi

RUN_ID=$1
shift
RESUME=false
MODEL=
for argument in "$@"; do
  case "$argument" in
    --resume) RESUME=true ;;
    Qwen3-8B|Gemma4-E4B) MODEL=$argument ;;
    *) echo "Unknown argument: $argument" >&2; exit 2 ;;
  esac
done

if [[ ! "$RUN_ID" =~ ^[A-Za-z0-9][A-Za-z0-9._-]{2,90}$ ]]; then
  echo "Unsafe RUN_ID: $RUN_ID" >&2
  exit 2
fi

ROOT=/home/ubuntu/v443_ov_causal
REPO_ROOT=$ROOT/repo
PYTHON=$ROOT/.venv/bin/python
SOURCE_RUN=/lambda/nfs/CoT-Non-thinking-v4/runs/run_20260731_v4_numeric_presentation_v3
NAMESPACE=/lambda/nfs/CoT-Non-thinking-v4/runs/v4_4_3_ov_causal
RUN_ROOT=$NAMESPACE/$RUN_ID
HF_CACHE=/lambda/nfs/CoT-Non-thinking-v4/hf-cache
LOCAL_LOG_ROOT=$ROOT/logs
LOCAL_LOG=$LOCAL_LOG_ROOT/${RUN_ID}.log

mkdir -p "$LOCAL_LOG_ROOT"
exec 9>"$ROOT/${RUN_ID}.campaign.lock"
if ! flock -n 9; then
  echo "Another launcher owns V4.4.3-Set RUN_ID=$RUN_ID" >&2
  exit 3
fi

exec > >(tee -a "$LOCAL_LOG") 2>&1
echo "[$(date -u +'%Y-%m-%dT%H:%M:%SZ')] V4.4.3-Set launcher start"
echo "run_root=$RUN_ROOT"
echo "source_run=$SOURCE_RUN (read-only contract)"
nvidia-smi --query-gpu=name,uuid,memory.total,driver_version --format=csv,noheader

export PYTHONPATH=$REPO_ROOT/src
export HF_HOME=$HF_CACHE
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1

ARGS=(
  --stage campaign
  --run-root "$RUN_ROOT"
  --source-run-root "$SOURCE_RUN"
  --output-namespace-root "$NAMESPACE"
  --config "$REPO_ROOT/configs/realistic_niah_v4_4_3_set.json"
  --v4-config "$REPO_ROOT/configs/realistic_niah_v4.json"
  --cache-dir "$HF_CACHE"
  --repo-root "$REPO_ROOT"
  --device-map auto
)
if [[ "$RESUME" == true ]]; then ARGS+=(--resume); fi
if [[ -n "$MODEL" ]]; then ARGS+=(--model "$MODEL"); fi

"$PYTHON" "$REPO_ROOT/scripts/run_realistic_niah_v4_4_3_set.py" "${ARGS[@]}"

if [[ -z "$MODEL" ]]; then
  "$PYTHON" "$REPO_ROOT/scripts/build_realistic_niah_v4_4_3_set_report.py" \
    --run-root "$RUN_ROOT" \
    --output "$RUN_ROOT/analysis/realistic_niah_v4_4_3_ov_set_causal_report.html"
fi

if [[ -d "$RUN_ROOT" ]]; then
  cp "$LOCAL_LOG" "$RUN_ROOT/launcher.log.tmp"
  mv "$RUN_ROOT/launcher.log.tmp" "$RUN_ROOT/launcher.log"
fi
echo "[$(date -u +'%Y-%m-%dT%H:%M:%SZ')] V4.4.3-Set launcher complete"
