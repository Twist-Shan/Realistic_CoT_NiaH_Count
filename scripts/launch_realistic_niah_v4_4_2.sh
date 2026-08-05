#!/usr/bin/env bash
set -Eeuo pipefail

: "${RUN_ROOT:?Set RUN_ROOT to the new append-only V4.4.2 directory.}"
: "${STIMULI:?Set STIMULI to the frozen V4 stimuli.jsonl.}"
: "${HF_CACHE:?Set HF_CACHE to the Hugging Face cache directory.}"

PYTHON_BIN="${V442_PYTHON:-python3}"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONFIG="${V442_CONFIG:-${REPO_ROOT}/configs/realistic_niah_v4_4_2.json}"
ANALYSIS_DEVICE="${V442_ANALYSIS_DEVICE:-cuda}"
LOG_DIR="${RUN_ROOT}/logs"
MODELS=(Qwen3-8B Gemma4-E4B)
MODES=(nonthinking native_thinking)
PROMPTS=(cue_present cue_absent)

mkdir -p "${RUN_ROOT}" "${HF_CACHE}" "${LOG_DIR}"
cd "${REPO_ROOT}"

run_logged() {
  local label="$1"
  shift
  echo "[v4.4.2] start ${label} $(date --iso-8601=seconds)"
  "$@" 2>&1 | tee -a "${LOG_DIR}/${label}.log"
  echo "[v4.4.2] complete ${label} $(date --iso-8601=seconds)"
}

# All eight cells are independently generated and teacher-replayed because the
# H100 worker cannot access the earlier V4.4 filestream.
for model in "${MODELS[@]}"; do
  for mode in "${MODES[@]}"; do
    for prompt in "${PROMPTS[@]}"; do
      label="generate_capture_${model}_${mode}_${prompt}"
      run_logged "${label}" \
        env PYTHONPATH=src "${PYTHON_BIN}" scripts/run_realistic_niah_v4_4_2.py \
        --stage generate-capture --stimuli "${STIMULI}" --config "${CONFIG}" \
        --output-dir "${RUN_ROOT}" --model "${model}" --mode "${mode}" \
        --prompt-variant "${prompt}" --cache-dir "${HF_CACHE}"
    done
  done
done

# This stage does not load a model. CUDA is the full-run default because QK
# reconstruction is compute-heavy; set V442_ANALYSIS_DEVICE=cpu only if needed.
run_logged analyze_existing \
  env PYTHONPATH=src "${PYTHON_BIN}" scripts/run_realistic_niah_v4_4_2.py \
  --stage analyze-existing --config "${CONFIG}" --output-dir "${RUN_ROOT}" \
  --analysis-device "${ANALYSIS_DEVICE}"
run_logged index \
  env PYTHONPATH=src "${PYTHON_BIN}" scripts/run_realistic_niah_v4_4_2.py \
  --stage index --config "${CONFIG}" --output-dir "${RUN_ROOT}"
run_logged aggregate \
  env PYTHONPATH=src "${PYTHON_BIN}" scripts/run_realistic_niah_v4_4_2.py \
  --stage aggregate --config "${CONFIG}" --output-dir "${RUN_ROOT}"
