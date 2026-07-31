#!/usr/bin/env bash
set -Eeuo pipefail

: "${RUN_ROOT:?Set RUN_ROOT to the immutable formal-run directory.}"
: "${HF_CACHE:?Set HF_CACHE to the Hugging Face cache directory.}"

V4_PYTHON="${V4_PYTHON:-python3}"
REPO_ROOT="$(
  cd "$(dirname "${BASH_SOURCE[0]}")/.."
  pwd
)"
CONFIG="${V4_CONFIG:-${REPO_ROOT}/configs/realistic_niah_v4.json}"
DATASET_DIR="${RUN_ROOT}/dataset"
STIMULI="${DATASET_DIR}/stimuli.jsonl"
LOG_DIR="${RUN_ROOT}/logs"
MODELS=(Qwen3-8B Gemma4-E4B)

mkdir -p "${RUN_ROOT}" "${HF_CACHE}" "${LOG_DIR}"
exec 9>"${RUN_ROOT}/formal_run.lock"
if ! flock -n 9; then
  echo "Another V4 formal launcher holds ${RUN_ROOT}/formal_run.lock" >&2
  exit 73
fi

STARTED_AT="$(date --iso-8601=seconds)"
ATTEMPT_ID="$(date -u +%Y%m%dT%H%M%SZ)"
STATUS_FILE="${RUN_ROOT}/formal_run_${ATTEMPT_ID}.status"

finish() {
  exit_code=$?
  {
    printf 'attempt_id=%s\n' "${ATTEMPT_ID}"
    printf 'started_at=%s\n' "${STARTED_AT}"
    printf 'finished_at=%s\n' "$(date --iso-8601=seconds)"
    printf 'exit_code=%s\n' "${exit_code}"
  } >"${STATUS_FILE}"
  if [[ "${exit_code}" -eq 0 ]]; then
    touch "${RUN_ROOT}/formal_run.complete"
  fi
}
trap finish EXIT

cd "${REPO_ROOT}"
if [[ -n "$(git status --porcelain)" ]]; then
  echo "Formal V4 launch requires a clean Git worktree." >&2
  git status --short >&2
  exit 65
fi

git rev-parse HEAD >"${RUN_ROOT}/code_commit.txt"
git status --short --branch >"${RUN_ROOT}/git_status.txt"
"${V4_PYTHON}" --version >"${RUN_ROOT}/python_version.txt" 2>&1
"${V4_PYTHON}" -m pip freeze >"${RUN_ROOT}/python_packages.txt"
nvidia-smi -q >"${RUN_ROOT}/nvidia_smi_q.txt"
cp "${CONFIG}" "${RUN_ROOT}/realistic_niah_v4.resolved.json"

run_logged() {
  local label="$1"
  shift
  echo "[formal] start ${label} $(date --iso-8601=seconds)"
  "$@" 2>&1 | tee -a "${LOG_DIR}/${label}.log"
  echo "[formal] complete ${label} $(date --iso-8601=seconds)"
}

if [[ ! -f "${DATASET_DIR}/audit.json" ]]; then
  run_logged freeze \
    env PYTHONPATH=src "${V4_PYTHON}" scripts/freeze_realistic_niah_v4.py \
    --config "${CONFIG}" \
    --output-dir "${DATASET_DIR}" \
    --cache-dir "${HF_CACHE}"
fi

env PYTHONPATH=src "${V4_PYTHON}" - "${DATASET_DIR}/audit.json" <<'PY'
import json
import pathlib
import sys

path = pathlib.Path(sys.argv[1])
payload = json.loads(path.read_text(encoding="utf-8"))
if payload.get("passed") is not True:
    raise SystemExit(f"Frozen V4 audit did not pass: {path}")
PY

for model in "${MODELS[@]}"; do
  run_logged "preflight_${model}" \
    env PYTHONPATH=src "${V4_PYTHON}" scripts/run_realistic_niah_v4.py \
    --stage preflight \
    --stimuli "${STIMULI}" \
    --config "${CONFIG}" \
    --output-dir "${RUN_ROOT}" \
    --model "${model}" \
    --cache-dir "${HF_CACHE}" \
    --forward-smoke \
    --repo-root "${REPO_ROOT}"
done

for model in "${MODELS[@]}"; do
  run_logged "representation_capture_${model}" \
    env PYTHONPATH=src "${V4_PYTHON}" scripts/run_realistic_niah_v4.py \
    --stage representation-capture \
    --stimuli "${STIMULI}" \
    --config "${CONFIG}" \
    --output-dir "${RUN_ROOT}" \
    --model "${model}" \
    --cache-dir "${HF_CACHE}" \
    --repo-root "${REPO_ROOT}"

  run_logged "representation_analyze_${model}" \
    env PYTHONPATH=src "${V4_PYTHON}" scripts/run_realistic_niah_v4.py \
    --stage representation-analyze \
    --stimuli "${STIMULI}" \
    --config "${CONFIG}" \
    --output-dir "${RUN_ROOT}" \
    --model "${model}" \
    --repo-root "${REPO_ROOT}"
done

for model in "${MODELS[@]}"; do
  run_logged "attention_${model}" \
    env PYTHONPATH=src "${V4_PYTHON}" scripts/run_realistic_niah_v4.py \
    --stage attention \
    --stimuli "${STIMULI}" \
    --config "${CONFIG}" \
    --output-dir "${RUN_ROOT}" \
    --model "${model}" \
    --cache-dir "${HF_CACHE}" \
    --repo-root "${REPO_ROOT}"
done

for model in "${MODELS[@]}"; do
  run_logged "ablation_${model}" \
    env PYTHONPATH=src "${V4_PYTHON}" scripts/run_realistic_niah_v4.py \
    --stage ablation \
    --stimuli "${STIMULI}" \
    --config "${CONFIG}" \
    --output-dir "${RUN_ROOT}" \
    --model "${model}" \
    --cache-dir "${HF_CACHE}" \
    --repo-root "${REPO_ROOT}"
done

for model in "${MODELS[@]}"; do
  run_logged "patching_${model}" \
    env PYTHONPATH=src "${V4_PYTHON}" scripts/run_realistic_niah_v4.py \
    --stage patching \
    --stimuli "${STIMULI}" \
    --config "${CONFIG}" \
    --output-dir "${RUN_ROOT}" \
    --model "${model}" \
    --cache-dir "${HF_CACHE}" \
    --repo-root "${REPO_ROOT}"
done
