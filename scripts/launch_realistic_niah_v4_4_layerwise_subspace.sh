#!/usr/bin/env bash
set -euo pipefail

RUN_ROOT="${RUN_ROOT:-/lambda/nfs/CoT-Non-thinking-v4/runs/v4_4_layerwise_subspace_20260809}"
CODE="${CODE:-$RUN_ROOT/code}"
PY="${PY:-/lambda/nfs/CoT-Non-thinking-v4/venv/bin/python}"
V4_CONFIG="${V4_CONFIG:-/lambda/nfs/CoT-Non-thinking-v4/repo/configs/realistic_niah_v4.json}"
STIMULI="${STIMULI:-/lambda/nfs/CoT-Non-thinking-v4/runs/run_20260731_v4_numeric_presentation_v3/dataset/stimuli.jsonl}"
PACKED="${PACKED:-/lambda/nfs/CoT-Non-thinking-v4/runs/v4_4_counter_channel_20260806/packed}"
DESIGN="${DESIGN:-$CODE/configs/realistic_niah_v4_4_layerwise_subspace.json}"
CACHE="${CACHE:-/lambda/nfs/CoT-Non-thinking-v4/hf-cache}"
STAGE="${1:-all}"
MODEL="${2:-}"

export PYTHONPATH="$CODE/src"
mkdir -p "$RUN_ROOT/raw/prompt_removal" "$RUN_ROOT/raw/transport" \
  "$RUN_ROOT/analysis" "$RUN_ROOT/logs"

run_maps() {
  "$PY" "$CODE/scripts/analyze_v446_layerwise_rotation.py" \
    --packed-root "$PACKED" --design-config "$DESIGN" \
    --output "$RUN_ROOT/analysis/layer_maps"
}

run_prompt() {
  local model="$1"
  "$PY" "$CODE/scripts/run_v446_layerwise_prompt_removal.py" \
    --v4-config "$V4_CONFIG" --design-config "$DESIGN" \
    --stimuli "$STIMULI" --packed-root "$PACKED" \
    --output "$RUN_ROOT/raw/prompt_removal/$model" \
    --models "$model" --cache-dir "$CACHE"
}

run_transport() {
  local model="$1"
  "$PY" "$CODE/scripts/run_v446_layerwise_transport_patch.py" \
    --v4-config "$V4_CONFIG" --design-config "$DESIGN" \
    --stimuli "$STIMULI" --layer-root "$PACKED/layers" \
    --output "$RUN_ROOT/raw/transport/$model" \
    --models "$model" --cache-dir "$CACHE"
}

run_analysis() {
  "$PY" "$CODE/scripts/analyze_v446_layerwise_prompt_removal.py" \
    "$RUN_ROOT/raw/prompt_removal" --design-config "$DESIGN" \
    --output "$RUN_ROOT/analysis/prompt_removal"
  "$PY" "$CODE/scripts/analyze_v446_layerwise_transport_patch.py" \
    "$RUN_ROOT/raw/transport" --design-config "$DESIGN" \
    --output "$RUN_ROOT/analysis/transport"
}

case "$STAGE" in
  maps) run_maps ;;
  prompt)
    test -n "$MODEL"
    run_prompt "$MODEL"
    ;;
  transport)
    test -n "$MODEL"
    run_transport "$MODEL"
    ;;
  analyze) run_analysis ;;
  all)
    run_maps
    run_prompt Qwen3-8B
    run_prompt Gemma4-E4B
    run_transport Qwen3-8B
    run_transport Gemma4-E4B
    run_analysis
    ;;
  *)
    echo "usage: $0 {maps|prompt MODEL|transport MODEL|analyze|all}" >&2
    exit 2
    ;;
esac
