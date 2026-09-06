#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 2 ]]; then
  echo "usage: $0 <Qwen3-8B|Gemma4-E4B> <gpu-index>" >&2
  exit 2
fi

MODEL=$1
GPU_INDEX=$2
case "$MODEL" in
  Qwen3-8B|Gemma4-E4B) ;;
  *) echo "unsupported model: $MODEL" >&2; exit 2 ;;
esac
if [[ ! "$GPU_INDEX" =~ ^[0-9]+$ ]]; then
  echo "gpu-index must be a non-negative integer" >&2
  exit 2
fi

ROOT=${V6_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}
PYTHON=${V6_PYTHON:-$ROOT/.venv/bin/python}
CACHE=${V6_CACHE:-$ROOT/.cache/huggingface}
RUN_BASE=${V6_RUN_BASE:-$ROOT/work/realistic_niah_v6}
SUPERVISOR=$ROOT/scripts/supervise_realistic_niah_v6_targeted_retrieval.sh
QUEUE_ROOT=$RUN_BASE/queue_logs

mkdir -p "$QUEUE_ROOT"
exec > >(tee -a "$QUEUE_ROOT/${MODEL}.log") 2>&1

wait_for_pass() {
  local marker=$1
  local label=$2
  echo "[$(date --iso-8601=seconds)] WAIT $label marker=$marker"
  while [[ ! -s "$marker" ]]; do
    sleep 30
  done
  if ! grep -qx 'PASS' "$marker"; then
    echo "$label marker is not PASS: $marker" >&2
    exit 1
  fi
  echo "[$(date --iso-8601=seconds)] READY $label"
}

run_phase() {
  local mode=$1
  local phase=$2
  local prompt_mode=enumeration_$mode
  local run_root=$RUN_BASE/$prompt_mode/$MODEL
  echo "[$(date --iso-8601=seconds)] START $MODEL $mode $phase"
  env \
    V6_ROOT="$ROOT" \
    V6_PYTHON="$PYTHON" \
    V6_CACHE="$CACHE" \
    V6_RUN_ROOT="$run_root" \
    V6_CUDA_VISIBLE_DEVICES="$GPU_INDEX" \
    HF_HUB_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1 \
    bash "$SUPERVISOR" "$mode" "$MODEL" "$phase"
  echo "[$(date --iso-8601=seconds)] PASS $MODEL $mode $phase"
}

for mode in index bullet; do
  wait_for_pass \
    "$RUN_BASE/enumeration_${mode}/$MODEL/replacement/discovery/discovery.COMPLETE" \
    "$MODEL $mode strict replacement quota"
  wait_for_pass \
    "$RUN_BASE/enumeration_${mode}/$MODEL/discovery-foundation-resolved.COMPLETE" \
    "$MODEL $mode resolved foundation"
  wait_for_pass \
    "$RUN_BASE/enumeration_${mode}/$MODEL/replacement/discovery_broad_k/k_selection_discovery.COMPLETE" \
    "$MODEL $mode true-source-coherent broad K panel"
done

run_phase index all
run_phase bullet all
printf 'PASS\n' >"$QUEUE_ROOT/${MODEL}.COMPLETE"
echo "[$(date --iso-8601=seconds)] COMPLETE $MODEL targeted-retrieval queue"
