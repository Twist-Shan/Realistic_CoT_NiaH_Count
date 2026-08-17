#!/usr/bin/env bash
set -Eeuo pipefail

: "${GENERATIONS_ROOT:?Set GENERATIONS_ROOT to the directory containing <model>/generations.jsonl.}"
: "${OUTPUT_ROOT:?Set OUTPUT_ROOT to a new geometry capture directory.}"
: "${HF_CACHE:?Set HF_CACHE to the Hugging Face cache directory.}"

V5_PYTHON="${V5_PYTHON:-python3}"
REPO_ROOT="$({ cd "$(dirname "${BASH_SOURCE[0]}")/.."; pwd; })"
CONFIG="${V5_CONFIG:-${REPO_ROOT}/configs/realistic_niah_v5.json}"
LOG_DIR="${OUTPUT_ROOT}/logs"
LOCK="${OUTPUT_ROOT}/geometry_full_panel.lock"
MODELS=(Qwen3-8B Gemma4-E4B)
RUNNING_SITES=(pre_city city_end city_unit_end item_end post_boundary)

mkdir -p "${LOG_DIR}" "${HF_CACHE}"
exec 9>"${LOCK}"
if ! flock -n 9; then
  echo "Another geometry capture holds ${LOCK}" >&2
  exit 73
fi

cd "${REPO_ROOT}"
git rev-parse HEAD >"${OUTPUT_ROOT}/code_commit.txt"
git status --short --branch >"${OUTPUT_ROOT}/git_status.txt"
"${V5_PYTHON}" --version >"${OUTPUT_ROOT}/python_version.txt" 2>&1
nvidia-smi -q >"${OUTPUT_ROOT}/nvidia_smi_q.txt"

for model in "${MODELS[@]}"; do
  generations="${GENERATIONS_ROOT}/${model}/generations.jsonl"
  model_output="${OUTPUT_ROOT}/running/${model}"
  answer_output="${OUTPUT_ROOT}/final/${model}"
  log="${LOG_DIR}/${model}.log"
  if [[ ! -f "${generations}" ]]; then
    echo "Missing generation archive: ${generations}" >&2
    exit 66
  fi
  echo "[v5 geometry] start model=${model} $(date --iso-8601=seconds)" | tee -a "${log}"
  env PYTHONPATH=src "${V5_PYTHON}" scripts/run_realistic_niah_v5.py capture \
    --config "${CONFIG}" \
    --model "${model}" \
    --cache-dir "${HF_CACHE}" \
    --device-map auto \
    --torch-dtype bfloat16 \
    --attention-backend sdpa \
    --generations "${generations}" \
    --output "${model_output}" \
    --site-kinds "${RUNNING_SITES[@]}" answer_query_v3 \
    --skip-span-pooling 2>&1 | tee -a "${log}"
  env PYTHONPATH=src "${V5_PYTHON}" scripts/audit_realistic_niah_v5_geometry_capture.py \
    --capture-index "${model_output}/capture_index.jsonl" \
    --output "${model_output}/geometry_capture_audit.json" 2>&1 | tee -a "${log}"
  env PYTHONPATH=src "${V5_PYTHON}" scripts/split_realistic_niah_v5_capture.py \
    --source-index "${model_output}/capture_index.jsonl" \
    --output "${answer_output}" \
    --site-kinds answer_query_v3 2>&1 | tee -a "${log}"
  echo "[v5 geometry] complete model=${model} $(date --iso-8601=seconds)" | tee -a "${log}"
done
