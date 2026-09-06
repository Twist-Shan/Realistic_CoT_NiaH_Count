#!/usr/bin/env bash
set -euo pipefail

[[ $# -eq 1 ]] || { echo "Usage: $0 RUN_ROOT" >&2; exit 2; }
[[ -n "${SLURM_PROCID:-}" ]] \
  || { echo "SLURM_PROCID is required" >&2; exit 2; }
[[ "${SLURM_PROCID}" == "0" || "${SLURM_PROCID}" == "1" ]] \
  || { echo "Expected worker index 0 or 1" >&2; exit 2; }

run_root="$(readlink -f "$1")"
repo="${REALISTIC_NIAH_REPO_ROOT:?Set REALISTIC_NIAH_REPO_ROOT}"
python_bin="${REALISTIC_NIAH_PYTHON:?Set REALISTIC_NIAH_PYTHON}"
hf_cache="${REALISTIC_NIAH_HF_CACHE:?Set REALISTIC_NIAH_HF_CACHE}"
model_label="${REALISTIC_NIAH_MODEL_LABEL:?Set REALISTIC_NIAH_MODEL_LABEL}"

printf 'model=%s worker=%s host=%s CUDA_VISIBLE_DEVICES=%s\n' \
  "${model_label}" "${SLURM_PROCID}" "$(hostname)" \
  "${CUDA_VISIBLE_DEVICES:-<unset>}"
cd "${repo}"
PYTHONPATH=src "${python_bin}" \
  scripts/run_realistic_niah_v3_3_long_context_worker.py \
  --model "${model_label}" \
  --worker-index "${SLURM_PROCID}" \
  --stimuli "${run_root}/dataset/stimuli.jsonl" \
  --run-root "${run_root}" \
  --cache-dir "${hf_cache}" \
  --repo-root "${repo}" \
  --preflight-timeout-seconds 3600
