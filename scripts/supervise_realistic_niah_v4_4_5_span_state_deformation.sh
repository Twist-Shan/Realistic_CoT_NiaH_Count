#!/usr/bin/env bash
set -euo pipefail

RUN_ROOT="${1:-/home/ubuntu/runs/nonthinking_v445_span_state_deformation_20260817}"
REPO="${REPO:-/home/ubuntu/Realistic_CoT_NiaH_Count}"
CODE_ROOT="${CODE_ROOT:-$RUN_ROOT/code}"
PYTHON="${PYTHON:-/home/ubuntu/venvs/rniah/bin/python}"
CAMPAIGN="${CAMPAIGN:-/lambda/nfs/CoT-Non-thinking-v4/runs/nonthinking_v445_8gpu_20260813}"
STIMULI="${STIMULI:-$CAMPAIGN/dataset/canonical_run_20260731_v4_numeric_presentation_v3_stimuli.jsonl}"
CACHE_DIR="${CACHE_DIR:-/lambda/nfs/CoT-Non-thinking-v4/hf-cache}"
EXPECTED_STIMULUS_SHA="da4dd86142eb8a07f9a7e53497efd3375184c8e68367d4db994370fcb331f090"

mkdir -p "$RUN_ROOT/formal" "$RUN_ROOT/logs" "$RUN_ROOT/locks"
exec 9>"$RUN_ROOT/locks/span_state_deformation.lock"
if ! flock -n 9; then
  echo "another span-state deformation supervisor owns the global lock" >&2
  exit 75
fi

test -x "$PYTHON"
test -f "$STIMULI"
test -f "$CODE_ROOT/configs/realistic_niah_v4_4_5_stimuli.json"
test -f "$CODE_ROOT/configs/realistic_niah_v4_4_5_span_state_deformation.json"
test -f "$CODE_ROOT/scripts/run_realistic_niah_v4_4_5_span_state_deformation.py"
test -f "$RUN_ROOT/code_manifest.sha256"

ACTUAL_STIMULUS_SHA="$(sha256sum "$STIMULI" | awk '{print $1}')"
if [[ "$ACTUAL_STIMULUS_SHA" != "$EXPECTED_STIMULUS_SHA" ]]; then
  echo "stimulus hash mismatch: $ACTUAL_STIMULUS_SHA" >&2
  exit 1
fi
ACTUAL_MANIFEST_SHA="$(sha256sum "$RUN_ROOT/code_manifest.sha256" | awk '{print $1}')"
if [[ -z "${EXPECTED_CODE_MANIFEST_SHA:-}" ]]; then
  echo "EXPECTED_CODE_MANIFEST_SHA is required" >&2
  exit 1
fi
if [[ "$ACTUAL_MANIFEST_SHA" != "$EXPECTED_CODE_MANIFEST_SHA" ]]; then
  echo "code manifest mismatch: $ACTUAL_MANIFEST_SHA != $EXPECTED_CODE_MANIFEST_SHA" >&2
  exit 1
fi

cd "$REPO"
export PYTHONPATH="$CODE_ROOT:$CODE_ROOT/src:$REPO/src${PYTHONPATH:+:$PYTHONPATH}"
export TOKENIZERS_PARALLELISM=false
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export CUDA_VISIBLE_DEVICES=0

for MODEL in Qwen3-8B Gemma4-E4B; do
  echo "START model=$MODEL utc=$(date -u +%FT%TZ)"
  "$PYTHON" "$CODE_ROOT/scripts/run_realistic_niah_v4_4_5_span_state_deformation.py" \
    --model "$MODEL" \
    --stimuli "$STIMULI" \
    --stimuli-config "$CODE_ROOT/configs/realistic_niah_v4_4_5_stimuli.json" \
    --experiment-config "$CODE_ROOT/configs/realistic_niah_v4_4_5_span_state_deformation.json" \
    --output-dir "$RUN_ROOT/formal" \
    --cache-dir "$CACHE_DIR" \
    --device-map auto \
    2>&1 | tee "$RUN_ROOT/logs/${MODEL}_formal.log"
  "$PYTHON" -c 'import json, pathlib, sys; p=pathlib.Path(sys.argv[1]); d=json.loads(p.read_text()); assert d["status"] == "PASS", d' "$RUN_ROOT/formal/$MODEL/complete.json"
  test -f "$RUN_ROOT/formal/$MODEL/.RUN_COMPLETE"
  echo "COMPLETE model=$MODEL utc=$(date -u +%FT%TZ)"
done

cat >"$RUN_ROOT/formal/supervisor_complete.json" <<EOF
{
  "schema_version": "realistic_niah_v4_4_5_span_state_deformation_supervisor_v1",
  "status": "PASS",
  "models": ["Qwen3-8B", "Gemma4-E4B"],
  "code_manifest_sha256": "$ACTUAL_MANIFEST_SHA",
  "stimulus_sha256": "$ACTUAL_STIMULUS_SHA",
  "completed_utc": "$(date -u +%FT%TZ)"
}
EOF
touch "$RUN_ROOT/formal/.BOTH_MODELS_RUN_COMPLETE"
