#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 || $# -gt 2 ]]; then
  echo "usage: $0 MODEL [RUN_ROOT]" >&2
  exit 2
fi

MODEL="$1"
RUN_ROOT="${2:-/home/ubuntu/runs/nonthinking_v445_exp23_noise_factorial_20260814}"
REPO="${REPO:-/home/ubuntu/Realistic_CoT_NiaH_Count}"
PYTHON="${PYTHON:-/home/ubuntu/venvs/rniah/bin/python}"
CAMPAIGN="${CAMPAIGN:-/lambda/nfs/CoT-Non-thinking-v4/runs/nonthinking_v445_8gpu_20260813}"
STIMULI="${STIMULI:-$CAMPAIGN/dataset/canonical_run_20260731_v4_numeric_presentation_v3_stimuli.jsonl}"
CACHE_DIR="${CACHE_DIR:-/lambda/nfs/CoT-Non-thinking-v4/hf-cache}"
EXPECTED_STIMULUS_SHA="da4dd86142eb8a07f9a7e53497efd3375184c8e68367d4db994370fcb331f090"

case "$MODEL" in
  Qwen3-8B|Gemma4-E4B) ;;
  *) echo "unsupported model: $MODEL" >&2; exit 2 ;;
esac

mkdir -p "$RUN_ROOT/logs" "$RUN_ROOT/locks"
exec 9>"$RUN_ROOT/locks/${MODEL}.lock"
if ! flock -n 9; then
  echo "another $MODEL noise-factorial supervisor owns the lock" >&2
  exit 75
fi
test -x "$PYTHON"
test -f "$STIMULI"
[[ "$(sha256sum "$STIMULI" | awk '{print $1}')" == "$EXPECTED_STIMULUS_SHA" ]]
if [[ -n "${EXPECTED_GIT_SHA:-}" ]]; then
  [[ "$(git -C "$REPO" rev-parse HEAD)" == "$EXPECTED_GIT_SHA" ]]
fi

cd "$REPO"
export PYTHONPATH="$REPO/src${PYTHONPATH:+:$PYTHONPATH}"
export TOKENIZERS_PARALLELISM=false

"$PYTHON" scripts/run_realistic_niah_v4_4_5_noise_factorial.py \
  --model "$MODEL" \
  --stimuli "$STIMULI" \
  --output-dir "$RUN_ROOT" \
  --cache-dir "$CACHE_DIR" \
  --device-map cuda

"$PYTHON" scripts/analyze_realistic_niah_v4_4_5_noise_factorial.py \
  --run-root "$RUN_ROOT" \
  --models "$MODEL" \
  --bootstrap-draws 10000

test -f "$RUN_ROOT/$MODEL/analysis_audit.json"
"$PYTHON" -c 'import json, pathlib, sys; p=pathlib.Path(sys.argv[1]); a=json.loads(p.read_text()); assert a.get("status")=="PASS"' \
  "$RUN_ROOT/$MODEL/analysis_audit.json"
touch "$RUN_ROOT/$MODEL/.RUN_AND_ANALYSIS_COMPLETE"
