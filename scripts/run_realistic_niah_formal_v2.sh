#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 || $# -gt 2 ]]; then
  echo "Usage: $0 RUN_ROOT [primary|glm4-control]" >&2
  exit 2
fi

run_root="$(readlink -f "$1")"
panel="${2:-primary}"
repo="${REALISTIC_NIAH_REPO_ROOT:-/lambda/nfs/Twist-CoT-Count-Multi-Model-v2/code/Realistic_CoT_NiaH_Count}"
cache="${REALISTIC_NIAH_HF_CACHE:-/lambda/nfs/Twist-CoT-Count-Multi-Model-v2/hf-cache}"
python_bin="${REALISTIC_NIAH_PYTHON:-/home/ubuntu/venvs/realistic-niah-vllm/bin/python}"
stimuli="${run_root}/dataset/stimuli.jsonl"
audit="${run_root}/orchestration/stimuli_audit.json"

case "${run_root}" in
  /lambda/nfs/Twist-CoT-Count-Multi-Model-v2/runs/realistic_niah_v2/eight_models_formal_*)
    ;;
  *)
    echo "Refusing unexpected run root: ${run_root}" >&2
    exit 2
    ;;
esac

test -s "${stimuli}"
test -s "${audit}"
test -d "${cache}"
test -x "${python_bin}"
test -z "$(git -C "${repo}" status --short)"
"${python_bin}" -c \
  'import json,sys; assert json.load(open(sys.argv[1], encoding="utf-8"))["passed"]' \
  "${audit}"
(
  cd "${run_root}/dataset"
  sha256sum -c SHA256SUMS
)

revision_for() {
  case "$1" in
    Qwen3-1.7B) echo "70d244cc86ccca08cf5af4e1e306ecf908b1ad5e" ;;
    Qwen3-4B) echo "1cfa9a7208912126459214e8b04321603b3df60c" ;;
    Qwen3-8B) echo "b968826d9c46dd6066d109eabc6255188de91218" ;;
    Qwen3-32B) echo "9216db5781bf21249d130ec9da846c4624c16137" ;;
    Gemma4-E4B) echo "ee0ef6023621cff504d758262d4e04895a5af4a2" ;;
    Gemma4-12B) echo "707f0a3b8a3c7ad586ed01e27eafbad8a27dd0f7" ;;
    DeepSeek-R1-0528-Qwen3-8B)
      echo "6e8885a6ff5c1dc5201574c8fd700323f23c25fa"
      ;;
    GLM-Z1-9B-0414) echo "b221b06fefb23ca320922cf6e68ab5f2fb82de81" ;;
    GLM-4-9B-0414) echo "645b8482494e31b6b752272bf7f7f273ef0f3caf" ;;
    *)
      echo "No frozen revision for model $1" >&2
      return 2
      ;;
  esac
}

engine_limits_for() {
  case "$1" in
    Qwen3-1.7B) echo "16 16" ;;
    Qwen3-4B) echo "12 12" ;;
    Qwen3-32B) echo "2 1" ;;
    Gemma4-12B) echo "6 6" ;;
    DeepSeek-R1-0528-Qwen3-8B|GLM-Z1-9B-0414) echo "6 4" ;;
    *) echo "8 8" ;;
  esac
}

case "${panel}" in
  primary)
    models=(
      Qwen3-1.7B
      Qwen3-4B
      Qwen3-8B
      Qwen3-32B
      Gemma4-E4B
      Gemma4-12B
      DeepSeek-R1-0528-Qwen3-8B
      GLM-Z1-9B-0414
    )
    prompt_modes="direct,enumeration_index,enumeration_bullet,native_thinking"
    output_base="${run_root}/models"
    status_file="${run_root}/orchestration/queue_status.primary.tsv"
    completion_file="${run_root}/orchestration/primary_completed_at_utc.txt"
    ;;
  glm4-control)
    models=(GLM-4-9B-0414)
    prompt_modes="direct,enumeration_index,enumeration_bullet"
    output_base="${run_root}/matched_controls"
    status_file="${run_root}/orchestration/queue_status.glm4_control.tsv"
    completion_file="${run_root}/orchestration/glm4_control_completed_at_utc.txt"
    ;;
  *)
    echo "Unknown panel: ${panel}" >&2
    exit 2
    ;;
esac

mkdir -p "${output_base}"
if [[ ! -e "${status_file}" ]]; then
  printf "model\tstatus\tupdated_at_utc\n" > "${status_file}"
fi

for model in "${models[@]}"; do
  model_prompt_modes="${prompt_modes}"
  case "${model}" in
    DeepSeek-R1-0528-Qwen3-8B|GLM-Z1-9B-0414)
      model_prompt_modes="native_thinking"
      ;;
  esac
  revision="$(revision_for "${model}")"
  read -r request_batch_size max_num_seqs < <(
    engine_limits_for "${model}"
  )
  model_output="${output_base}/${model}/main"
  log_file="${run_root}/orchestration/${panel}.${model}.log"
  timestamp="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  printf "%s\trunning\t%s\n" "${model}" "${timestamp}" >> "${status_file}"

  if (
    cd "${repo}"
    env \
      CUDA_VISIBLE_DEVICES=0 \
      PATH="/home/ubuntu/venvs/realistic-niah-vllm/bin:${PATH}" \
      PYTHONPATH=src \
      TOKENIZERS_PARALLELISM=false \
      "${python_bin}" scripts/run_realistic_niah.py \
        --stimuli "${stimuli}" \
        --output-dir "${model_output}" \
        --model "${model}" \
        --revision "${revision}" \
        --prompt-modes "${model_prompt_modes}" \
        --query-layout cue_before_query_after \
        --cache-dir "${cache}" \
        --repo-root "${repo}" \
        --tensor-parallel-size 1 \
        --max-model-len 32768 \
        --gpu-memory-utilization 0.90 \
        --max-num-seqs "${max_num_seqs}" \
        --request-batch-size "${request_batch_size}" \
        --require-clean-git
  ) > "${log_file}" 2>&1; then
    timestamp="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
    printf "%s\tcompleted\t%s\n" \
      "${model}" "${timestamp}" >> "${status_file}"
  else
    exit_code=$?
    timestamp="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
    printf "%s\tfailed(%s)\t%s\n" \
      "${model}" "${exit_code}" "${timestamp}" >> "${status_file}"
    exit "${exit_code}"
  fi
done

date -u +%Y-%m-%dT%H:%M:%SZ > "${completion_file}"
