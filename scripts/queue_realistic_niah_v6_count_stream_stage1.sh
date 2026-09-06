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

ROOT=${V6_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}
PYTHON=${V6_PYTHON:-$ROOT/.venv/bin/python}
CACHE=${V6_CACHE:-$ROOT/.cache/huggingface}
RUN_BASE=${V6_RUN_BASE:-$ROOT/work/realistic_niah_v6}
SUPERVISOR=$ROOT/scripts/supervise_realistic_niah_v6_count_stream_stage1.sh
QUEUE_ROOT=$RUN_BASE/queue_logs
mkdir -p "$QUEUE_ROOT"
exec > >(tee -a "$QUEUE_ROOT/${MODEL}_count_stream_stage1.log") 2>&1

wait_for_pass() {
  local marker=$1
  local label=$2
  echo "[$(date --iso-8601=seconds)] WAIT $label marker=$marker"
  while [[ ! -s "$marker" ]]; do sleep 30; done
  grep -qx PASS "$marker" || { echo "$label is not PASS" >&2; exit 1; }
  echo "[$(date --iso-8601=seconds)] READY $label"
}

wait_for_pass "$QUEUE_ROOT/$MODEL.COMPLETE" "$MODEL targeted retrieval"
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

for mode in index bullet; do
  echo "[$(date --iso-8601=seconds)] START $MODEL $mode count-stream stage-1"
  env V6_ROOT="$ROOT" V6_PYTHON="$PYTHON" V6_CACHE="$CACHE" \
    V6_RUN_BASE="$RUN_BASE" HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
    bash "$SUPERVISOR" "$mode" "$MODEL" "$GPU_INDEX"
  echo "[$(date --iso-8601=seconds)] PASS $MODEL $mode count-stream stage-1"
done

printf 'PASS\n' >"$QUEUE_ROOT/${MODEL}_count_stream_stage1.COMPLETE"
