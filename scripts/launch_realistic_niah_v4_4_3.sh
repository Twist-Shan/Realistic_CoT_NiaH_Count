#!/usr/bin/env bash
set -Eeuo pipefail

usage() {
  cat <<'EOF'
Usage:
  launch_realistic_niah_v4_4_3.sh RUN_ID [--resume] [MODEL]

MODEL may be Qwen3-8B or Gemma4-E4B.  Omitting MODEL runs both models and the
final analysis/audit.  The launcher reads the frozen V4.4 source run and writes
only below runs/v4_4_3_ov_causal/RUN_ID.
EOF
}

if [[ $# -lt 1 ]]; then
  usage >&2
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

if [[ ! "$RUN_ID" =~ ^[A-Za-z0-9][A-Za-z0-9._-]{2,80}$ ]]; then
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
  echo "Another local launcher owns V4.4.3 RUN_ID=$RUN_ID" >&2
  exit 3
fi

exec > >(tee -a "$LOCAL_LOG") 2>&1
echo "[$(date -u +'%Y-%m-%dT%H:%M:%SZ')] V4.4.3 launcher start"
echo "run_root=$RUN_ROOT"
echo "source_run=$SOURCE_RUN (read-only contract)"
echo "hostname=$(hostname)"
nvidia-smi --query-gpu=name,uuid,memory.total,driver_version --format=csv,noheader

if [[ ! -x "$PYTHON" ]]; then
  echo "Missing V4.4.3 virtualenv: $PYTHON" >&2
  exit 2
fi
if [[ ! -s "$SOURCE_RUN/dataset/stimuli.jsonl" ]]; then
  echo "Frozen V4.4 source is unavailable: $SOURCE_RUN" >&2
  exit 2
fi

export PYTHONPATH=$REPO_ROOT/src
export HF_HOME=$HF_CACHE
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1

ARGS=(
  --stage campaign
  --run-root "$RUN_ROOT"
  --source-run-root "$SOURCE_RUN"
  --output-namespace-root "$NAMESPACE"
  --config "$REPO_ROOT/configs/realistic_niah_v4_4_3.json"
  --v4-config "$REPO_ROOT/configs/realistic_niah_v4.json"
  --cache-dir "$HF_CACHE"
  --repo-root "$REPO_ROOT"
  --device-map auto
)
if [[ "$RESUME" == true ]]; then
  ARGS+=(--resume)
fi
if [[ -n "$MODEL" ]]; then
  ARGS+=(--model "$MODEL")
fi

"$PYTHON" "$REPO_ROOT/scripts/run_realistic_niah_v4_4_3.py" "${ARGS[@]}"

if [[ -d "$RUN_ROOT" ]]; then
  cp "$LOCAL_LOG" "$RUN_ROOT/launcher.log.tmp"
  mv "$RUN_ROOT/launcher.log.tmp" "$RUN_ROOT/launcher.log"
fi
echo "[$(date -u +'%Y-%m-%dT%H:%M:%SZ')] V4.4.3 launcher complete"
