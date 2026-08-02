#!/usr/bin/env bash
set -Eeuo pipefail

: "${RUN_ROOT:?Set RUN_ROOT to the existing immutable V4 run directory.}"
: "${HF_CACHE:?Set HF_CACHE to the Hugging Face cache directory.}"

V4_PYTHON="${V4_PYTHON:-python3}"
REPO_ROOT="$(
  cd "$(dirname "${BASH_SOURCE[0]}")/.."
  pwd
)"
CONFIG="${V4_CONFIG:-${REPO_ROOT}/configs/realistic_niah_v4.json}"
STIMULI="${RUN_ROOT}/dataset/stimuli.jsonl"
LOG_DIR="${RUN_ROOT}/logs"
STATUS="${RUN_ROOT}/answer_query_all_layers.status"
COMPLETE="${RUN_ROOT}/answer_query_all_layers_v1.complete"
LOCK="${RUN_ROOT}/answer_query_all_layers_v1.lock"
MODELS=(Qwen3-8B Gemma4-E4B)

mkdir -p "${LOG_DIR}" "${HF_CACHE}"
exec 9>"${LOCK}"
if ! flock -n 9; then
  echo "Another answer-query all-layer launcher holds ${LOCK}" >&2
  exit 73
fi

STARTED_AT="$(date --iso-8601=seconds)"
finish() {
  exit_code=$?
  {
    printf 'profile=answer_query_all_layers_v1\n'
    printf 'started_at=%s\n' "${STARTED_AT}"
    printf 'finished_at=%s\n' "$(date --iso-8601=seconds)"
    printf 'exit_code=%s\n' "${exit_code}"
  } >"${STATUS}"
  if [[ "${exit_code}" -eq 0 ]]; then
    touch "${COMPLETE}"
  fi
}
trap finish EXIT

cd "${REPO_ROOT}"
if [[ -n "$(git status --porcelain)" ]]; then
  echo "Answer-query capture requires a clean Git worktree." >&2
  git status --short >&2
  exit 65
fi
if [[ ! -f "${STIMULI}" ]]; then
  echo "Frozen stimuli are missing: ${STIMULI}" >&2
  exit 66
fi

DISCOVERY_SEEDS="$(${V4_PYTHON} -c 'import json,sys; print(",".join(str(value) for value in json.load(open(sys.argv[1], encoding="utf-8"))["discovery_seeds"]))' "${CONFIG}")"
VARIANTS="v4.1,v4.2,v4.3,v4.4"
COUNTS="1,2,3,4,5,6,7,8,9,10"

git rev-parse HEAD >"${RUN_ROOT}/answer_query_all_layers_code_commit.txt"
git status --short --branch >"${RUN_ROOT}/answer_query_all_layers_git_status.txt"
"${V4_PYTHON}" --version >"${RUN_ROOT}/answer_query_all_layers_python_version.txt" 2>&1
nvidia-smi -q >"${RUN_ROOT}/answer_query_all_layers_nvidia_smi_q.txt"

for model in "${MODELS[@]}"; do
  log="${LOG_DIR}/answer_query_all_layers_v1_${model}.log"
  echo "[answer-query-all-layers] start model=${model} $(date --iso-8601=seconds)" | tee -a "${log}"
  env PYTHONPATH=src "${V4_PYTHON}" scripts/run_realistic_niah_v4.py \
    --stage answer-query-representation-capture \
    --stimuli "${STIMULI}" \
    --config "${CONFIG}" \
    --output-dir "${RUN_ROOT}" \
    --model "${model}" \
    --answer-format numeric \
    --cache-dir "${HF_CACHE}" \
    --device-map auto \
    --variants "${VARIANTS}" \
    --seeds "${DISCOVERY_SEEDS}" \
    --counts "${COUNTS}" \
    --repo-root "${REPO_ROOT}" 2>&1 | tee -a "${log}"
  echo "[answer-query-all-layers] complete model=${model} $(date --iso-8601=seconds)" | tee -a "${log}"
done

