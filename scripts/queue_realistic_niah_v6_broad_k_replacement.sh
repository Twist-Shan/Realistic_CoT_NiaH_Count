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
BASE_POLICY=$ROOT/configs/realistic_niah_v6_replacement_policy.json
COHERENT_POLICY=$ROOT/configs/realistic_niah_v6_coherent_broad_replacement_policy.json
RUNNER=$ROOT/scripts/run_realistic_niah_v6_broad_panel_replacement.py
QUEUE_ROOT=$RUN_BASE/queue_logs
mkdir -p "$QUEUE_ROOT"
exec > >(tee -a "$QUEUE_ROOT/${MODEL}_broad_k_replacement.log") 2>&1

wait_for_pass() {
  local marker=$1
  local label=$2
  echo "[$(date --iso-8601=seconds)] WAIT $label marker=$marker"
  while [[ ! -s "$marker" ]]; do sleep 30; done
  grep -qx PASS "$marker" || { echo "$label is not PASS" >&2; exit 1; }
  echo "[$(date --iso-8601=seconds)] READY $label"
}

for mode in index bullet; do
  prompt_mode=enumeration_$mode
  model_root=$RUN_BASE/$prompt_mode/$MODEL
  wait_for_pass \
    "$model_root/replacement/discovery/discovery.COMPLETE" \
    "$MODEL $mode cell-resolved discovery quota"
  wait_for_pass \
    "$model_root/discovery-foundation-resolved.COMPLETE" \
    "$MODEL $mode resolved foundation"
  echo "[$(date --iso-8601=seconds)] START $MODEL $mode coherent broad K panel"
  env CUDA_VISIBLE_DEVICES="$GPU_INDEX" HF_HUB_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1 "$PYTHON" "$RUNNER" \
    --v6-config "$ROOT/configs/realistic_niah_v6_${prompt_mode}.json" \
    --mechanism-config \
      "$ROOT/configs/realistic_niah_v6_${prompt_mode}_count_stream_dev.json" \
    --replacement-policy "$BASE_POLICY" \
    --coherent-broad-policy "$COHERENT_POLICY" \
    --replacement-stimuli "$REPLACEMENT_POOL/stimuli.jsonl" \
    --base-cohort-registry \
      "$model_root/replacement/discovery/selected_cells.jsonl" \
    --model "$MODEL" --phase k_selection_discovery \
    --generation-root "$model_root/generation" \
    --output "$model_root/replacement/discovery_broad_k" \
    --cache-dir "$CACHE" --device-map auto --torch-dtype bfloat16 \
    --attention-backend sdpa
  echo "[$(date --iso-8601=seconds)] PASS $MODEL $mode coherent broad K panel"
done

printf 'PASS\n' >"$QUEUE_ROOT/${MODEL}_broad_k_replacement.COMPLETE"
echo "[$(date --iso-8601=seconds)] COMPLETE $MODEL coherent broad K panels"
