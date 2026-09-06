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
REPLACEMENT_POOL=${V6_REPLACEMENT_POOL:-$RUN_BASE/replacement_seed_pool}
SUPERVISOR=$ROOT/scripts/supervise_realistic_niah_v6_enumeration.sh
QUEUE_ROOT=$RUN_BASE/queue_logs
mkdir -p "$QUEUE_ROOT"
exec > >(tee -a "$QUEUE_ROOT/${MODEL}_discovery_replacement.log") 2>&1

wait_for_pass() {
  local marker=$1
  local label=$2
  echo "[$(date --iso-8601=seconds)] WAIT $label marker=$marker"
  while [[ ! -s "$marker" ]]; do sleep 30; done
  grep -qx PASS "$marker" || { echo "$label is not PASS" >&2; exit 1; }
  echo "[$(date --iso-8601=seconds)] READY $label"
}

run_phase() {
  local mode=$1
  local phase=$2
  local run_root=$RUN_BASE/enumeration_${mode}/$MODEL
  echo "[$(date --iso-8601=seconds)] START $MODEL $mode $phase"
  env V6_ROOT="$ROOT" V6_PYTHON="$PYTHON" V6_CACHE="$CACHE" \
    V6_RUN_ROOT="$run_root" V6_REPLACEMENT_POOL="$REPLACEMENT_POOL" \
    V6_CUDA_VISIBLE_DEVICES="$GPU_INDEX" HF_HUB_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1 bash "$SUPERVISOR" "$mode" "$MODEL" "$phase"
  echo "[$(date --iso-8601=seconds)] PASS $MODEL $mode $phase"
}

echo "[$(date --iso-8601=seconds)] WAIT frozen-amendment replacement seed pool"
while [[ ! -s "$REPLACEMENT_POOL/stimuli.jsonl" || \
         ! -s "$REPLACEMENT_POOL/manifest.json" ]]; do
  sleep 30
done
echo "[$(date --iso-8601=seconds)] READY frozen-amendment replacement seed pool"
for mode in index bullet; do
  wait_for_pass \
    "$RUN_BASE/enumeration_${mode}/$MODEL/discovery-foundation.COMPLETE" \
    "$MODEL $mode original foundation"
done

for mode in index bullet; do
  run_phase "$mode" discovery-supplement
  run_phase "$mode" discovery-foundation-resolved
done

printf 'PASS\n' >"$QUEUE_ROOT/${MODEL}_discovery_replacement.COMPLETE"
echo "[$(date --iso-8601=seconds)] COMPLETE $MODEL resolved discovery cohorts"
