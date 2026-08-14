#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 || $# -gt 2 ]]; then
  echo "usage: $0 MODEL [RUN_ROOT]" >&2
  exit 2
fi

MODEL="$1"
RUN_ROOT="${2:-/home/ubuntu/runs/nonthinking_v445_exp19_serial_mediation_20260814}"
REPO="${REPO:-/home/ubuntu/Realistic_CoT_NiaH_Count}"
PYTHON="${PYTHON:-/home/ubuntu/venvs/rniah/bin/python}"
CAMPAIGN="${CAMPAIGN:-/lambda/nfs/CoT-Non-thinking-v4/runs/nonthinking_v445_8gpu_20260813}"
PACKED="${PACKED:-/lambda/nfs/CoT-Non-thinking-v4/runs/v4_4_counter_channel_20260806/packed}"
STIMULI="${STIMULI:-$CAMPAIGN/dataset/canonical_run_20260731_v4_numeric_presentation_v3_stimuli.jsonl}"
RETRIEVAL_BASIS="${RETRIEVAL_BASIS:-$CAMPAIGN/analysis/retrieval_geometry/retrieval_bases.pt}"
CACHE_DIR="${CACHE_DIR:-/lambda/nfs/CoT-Non-thinking-v4/hf-cache}"
EXPECTED_STIMULUS_SHA="da4dd86142eb8a07f9a7e53497efd3375184c8e68367d4db994370fcb331f090"

case "$MODEL" in
  Qwen3-8B|Gemma4-E4B) ;;
  *) echo "unsupported model: $MODEL" >&2; exit 2 ;;
esac

mkdir -p "$RUN_ROOT/logs" "$RUN_ROOT/locks"
LOCK="$RUN_ROOT/locks/${MODEL}.lock"
exec 9>"$LOCK"
if ! flock -n 9; then
  echo "another $MODEL serial-mediation supervisor owns $LOCK" >&2
  exit 75
fi

test -x "$PYTHON"
test -f "$STIMULI"
test -f "$RETRIEVAL_BASIS"
test -f "$PACKED/layers/${MODEL}__answer_query__L$(
  [[ "$MODEL" == Qwen3-8B ]] && printf '29' || printf '37'
).npz"

ACTUAL_STIMULUS_SHA="$(sha256sum "$STIMULI" | awk '{print $1}')"
if [[ "$ACTUAL_STIMULUS_SHA" != "$EXPECTED_STIMULUS_SHA" ]]; then
  echo "stimulus hash mismatch: $ACTUAL_STIMULUS_SHA" >&2
  exit 1
fi
if [[ -n "${EXPECTED_GIT_SHA:-}" ]]; then
  ACTUAL_GIT_SHA="$(git -C "$REPO" rev-parse HEAD)"
  if [[ "$ACTUAL_GIT_SHA" != "$EXPECTED_GIT_SHA" ]]; then
    echo "git SHA mismatch: $ACTUAL_GIT_SHA != $EXPECTED_GIT_SHA" >&2
    exit 1
  fi
fi

cd "$REPO"
export PYTHONPATH="$REPO/src${PYTHONPATH:+:$PYTHONPATH}"
export TOKENIZERS_PARALLELISM=false

"$PYTHON" scripts/run_realistic_niah_v4_4_5_serial_mediation.py \
  --model "$MODEL" \
  --stimuli "$STIMULI" \
  --experiment-config configs/realistic_niah_v4_4_5_serial_mediation.json \
  --retrieval-basis "$RETRIEVAL_BASIS" \
  --answer-packed-root "$PACKED" \
  --output-dir "$RUN_ROOT" \
  --cache-dir "$CACHE_DIR" \
  --device-map cuda

test -f "$RUN_ROOT/$MODEL/complete.json"
touch "$RUN_ROOT/$MODEL/.RUN_COMPLETE"
