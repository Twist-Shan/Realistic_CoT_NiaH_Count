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
STATUS="${RUN_ROOT}/prompt_counter_attention.status"
COMPLETE="${RUN_ROOT}/prompt_counter_attention_v1.complete"
LOCK="${RUN_ROOT}/prompt_counter_attention_v1.lock"
MODELS=(Qwen3-8B Gemma4-E4B)

mkdir -p "${LOG_DIR}" "${HF_CACHE}"
exec 9>"${LOCK}"
if ! flock -n 9; then
  echo "Another prompt-counter attention launcher holds ${LOCK}" >&2
  exit 73
fi

STARTED_AT="$(date --iso-8601=seconds)"
finish() {
  exit_code=$?
  {
    printf 'profile=prompt_counter_attention_v1\n'
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
  echo "Prompt-counter attention capture requires a clean Git worktree." >&2
  git status --short >&2
  exit 65
fi
if [[ ! -f "${STIMULI}" ]]; then
  echo "Frozen stimuli are missing: ${STIMULI}" >&2
  exit 66
fi

ALL_SEEDS="$(${V4_PYTHON} -c 'import json,sys; print(",".join(str(value) for value in json.load(open(sys.argv[1], encoding="utf-8"))["seeds"]))' "${CONFIG}")"
VARIANTS="v4.1,v4.2,v4.3,v4.4"

git rev-parse HEAD >"${RUN_ROOT}/prompt_counter_attention_code_commit.txt"
git status --short --branch >"${RUN_ROOT}/prompt_counter_attention_git_status.txt"
"${V4_PYTHON}" --version >"${RUN_ROOT}/prompt_counter_attention_python_version.txt" 2>&1
nvidia-smi -q >"${RUN_ROOT}/prompt_counter_attention_nvidia_smi_q.txt"

for model in "${MODELS[@]}"; do
  log="${LOG_DIR}/prompt_counter_attention_v1_${model}.log"
  echo "[prompt-counter-attention] start model=${model} $(date --iso-8601=seconds)" | tee -a "${log}"
  env PYTHONPATH=src "${V4_PYTHON}" scripts/run_realistic_niah_v4.py \
    --stage prompt-counter-attention-capture \
    --stimuli "${STIMULI}" \
    --config "${CONFIG}" \
    --output-dir "${RUN_ROOT}" \
    --model "${model}" \
    --answer-format numeric \
    --cache-dir "${HF_CACHE}" \
    --device-map auto \
    --variants "${VARIANTS}" \
    --seeds "${ALL_SEEDS}" \
    --counts 10 \
    --repo-root "${REPO_ROOT}" 2>&1 | tee -a "${log}"
  echo "[prompt-counter-attention] complete model=${model} $(date --iso-8601=seconds)" | tee -a "${log}"
done
