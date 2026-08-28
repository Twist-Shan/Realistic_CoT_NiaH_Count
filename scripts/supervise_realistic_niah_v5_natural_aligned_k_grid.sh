#!/usr/bin/env bash
set -euo pipefail

ROOT="${1:-/home/ubuntu/same_site_progress_20260827}"
PYTHON_BIN="${PYTHON_BIN:-/lambda/nfs/CoT-Native-thinking-v5/venv/bin/python}"
CACHE_DIR="${CACHE_DIR:-/lambda/nfs/CoT-Native-thinking-v5/cache/huggingface}"
GENERATIONS="${GENERATIONS:-${ROOT}/work/input/n10/discovery_rows_first_pass_noindex_v5.jsonl}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${ROOT}/work/output/n10_kgrid5_v1}"
SEEDS="${SEEDS:-}"
MAX_SEEDS="${MAX_SEEDS:-5}"
GPU0_DONORS="${GPU0_DONORS:-2 3 4 5}"
GPU1_DONORS="${GPU1_DONORS:-7 8 9}"

mkdir -p "${OUTPUT_ROOT}"

selection_args=(--max-seeds "${MAX_SEEDS}")
if [[ -n "${SEEDS}" ]]; then
  read -r -a selected_seeds <<<"${SEEDS}"
  selection_args=(--seeds "${selected_seeds[@]}")
fi
read -r -a gpu0_donors <<<"${GPU0_DONORS}"
read -r -a gpu1_donors <<<"${GPU1_DONORS}"

run_group() {
  local gpu="$1"
  shift
  local donor receiver output log
  for donor in "$@"; do
    receiver=$((donor - 1))
    output="${OUTPUT_ROOT}/k${donor}"
    log="${OUTPUT_ROOT}/k${donor}.log"
    if [[ -f "${output}/manifest.json" ]] && grep -q '"status": "PASS"' "${output}/manifest.json"; then
      echo "[k-grid] skip completed k=${donor}" | tee -a "${log}"
      continue
    fi
    echo "[k-grid] start gpu=${gpu} j=${receiver} k=${donor}" | tee "${log}"
    (
      cd "${ROOT}"
      CUDA_VISIBLE_DEVICES="${gpu}" "${PYTHON_BIN}" \
        scripts/run_realistic_niah_v5_natural_aligned_progress_transplant.py \
        --model Qwen3-8B \
        --cache-dir "${CACHE_DIR}" \
        --device-map auto \
        --attention-backend sdpa \
        --prefill-chunk-size 256 \
        --generations "${GENERATIONS}" \
        --gold-count 10 \
        --receiver-occurrence "${receiver}" \
        --donor-occurrence "${donor}" \
        --layers 31 \
        --conditions receiver_self native_donor donor_to_receiver \
        --generation-conditions receiver_self donor_to_receiver \
        --max-new-tokens 256 \
        "${selection_args[@]}" \
        --output "${output}"
    ) >>"${log}" 2>&1
    echo "[k-grid] complete gpu=${gpu} k=${donor}" | tee -a "${log}"
  done
}

run_group 0 "${gpu0_donors[@]}" >"${OUTPUT_ROOT}/gpu0_supervisor.log" 2>&1 &
gpu0_pid=$!
run_group 1 "${gpu1_donors[@]}" >"${OUTPUT_ROOT}/gpu1_supervisor.log" 2>&1 &
gpu1_pid=$!

status=0
wait "${gpu0_pid}" || status=1
wait "${gpu1_pid}" || status=1
if [[ "${status}" -eq 0 ]]; then
  printf 'PASS\n' >"${OUTPUT_ROOT}/GRID_COMPLETE"
else
  printf 'FAIL\n' >"${OUTPUT_ROOT}/GRID_FAILED"
fi
exit "${status}"
