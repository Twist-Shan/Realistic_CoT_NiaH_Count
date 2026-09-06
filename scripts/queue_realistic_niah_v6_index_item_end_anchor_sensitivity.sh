#!/usr/bin/env bash
set -euo pipefail

ROOT=${V6_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}
PYTHON=${V6_PYTHON:-$ROOT/.venv/bin/python}
CACHE=${V6_CACHE:-$ROOT/.cache/huggingface}
RUN_BASE=${V6_RUN_BASE:-$ROOT/work/realistic_niah_v6}
GPU_INDEX=${V6_ITEM_END_GPU_INDEX:-0}
QUEUE_ROOT=$RUN_BASE/queue_logs
LOG=$QUEUE_ROOT/index_item_end_anchor_sensitivity.log
PRIMARY_GPU_RELEASE=$QUEUE_ROOT/Qwen3-8B_confirmation.COMPLETE
COMPLETE=$QUEUE_ROOT/index_item_end_anchor_sensitivity.COMPLETE

mkdir -p "$QUEUE_ROOT"
exec > >(tee -a "$LOG") 2>&1

echo "[$(date --iso-8601=seconds)] WAIT primary GPU release marker=$PRIMARY_GPU_RELEASE"
while [[ ! -s "$PRIMARY_GPU_RELEASE" ]]; do sleep 30; done
grep -qx PASS "$PRIMARY_GPU_RELEASE" || {
  echo "Qwen primary confirmation marker is not PASS" >&2
  exit 1
}
echo "[$(date --iso-8601=seconds)] READY GPU $GPU_INDEX for exploratory sensitivity"

common_env=(env V6_ROOT="$ROOT" V6_PYTHON="$PYTHON" V6_CACHE="$CACHE"
  V6_RUN_BASE="$RUN_BASE" HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1)

for model in Qwen3-8B Gemma4-E4B; do
  echo "[$(date --iso-8601=seconds)] START $model index item-end sensitivity"
  "${common_env[@]}" bash \
    "$ROOT/scripts/supervise_realistic_niah_v6_index_item_end_anchor_sensitivity.sh" \
    "$model" "$GPU_INDEX"
  echo "[$(date --iso-8601=seconds)] PASS $model index item-end sensitivity"
done

printf 'PASS\n' >"$COMPLETE"
echo "[$(date --iso-8601=seconds)] COMPLETE V6 index item-end anchor sensitivity"
