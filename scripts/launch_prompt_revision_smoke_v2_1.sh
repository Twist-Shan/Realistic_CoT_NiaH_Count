#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 1 ]]; then
  echo "Usage: $0 RUN_ROOT" >&2
  exit 2
fi

run_root="$(readlink -f "$1")"
repo="${REALISTIC_NIAH_REPO_ROOT:-/lambda/nfs/Twist-CoT-Count-Multi-Model-v2/code/Realistic_CoT_NiaH_Count_v2_1}"
cache="${REALISTIC_NIAH_HF_CACHE:-/lambda/nfs/Twist-CoT-Count-Multi-Model-v2/hf-cache}"
python_bin="${REALISTIC_NIAH_PYTHON:-/home/ubuntu/venvs/realistic-niah-vllm/bin/python}"
stimuli="${run_root}/dataset/stimuli.jsonl"
commit_file="${run_root}/orchestration/git_commit.txt"
expected_stimuli_sha="b739122c96adf73ec6df4abe0266af239a026b4de6f09f309933231f604c7f71"

case "${run_root}" in
  /lambda/nfs/Twist-CoT-Count-Multi-Model-v2/runs/realistic_niah_v2_1/prompt_revision_smoke_*)
    ;;
  *)
    echo "Refusing unexpected run root: ${run_root}" >&2
    exit 2
    ;;
esac

test -s "${stimuli}"
test -s "${commit_file}"
test -d "${cache}"
test -x "${python_bin}"
test -z "$(git -C "${repo}" status --short)"
expected_commit="$(tr -d '\r\n' < "${commit_file}")"
test "$(git -C "${repo}" rev-parse HEAD)" = "${expected_commit}"
test "$(sha256sum "${stimuli}" | awk '{print $1}')" = "${expected_stimuli_sha}"

revision_for() {
  case "$1" in
    Qwen3-1.7B) echo "70d244cc86ccca08cf5af4e1e306ecf908b1ad5e" ;;
    Qwen3-4B) echo "1cfa9a7208912126459214e8b04321603b3df60c" ;;
    Qwen3-8B) echo "b968826d9c46dd6066d109eabc6255188de91218" ;;
    Qwen3-32B) echo "9216db5781bf21249d130ec9da846c4624c16137" ;;
    Gemma4-E4B) echo "ee0ef6023621cff504d758262d4e04895a5af4a2" ;;
    Gemma4-12B) echo "707f0a3b8a3c7ad586ed01e27eafbad8a27dd0f7" ;;
    GLM-4-9B-0414) echo "645b8482494e31b6b752272bf7f7f273ef0f3caf" ;;
    *)
      echo "No frozen revision for model $1" >&2
      return 2
      ;;
  esac
}

launch() {
  local gpu="$1"
  local model="$2"
  local modes="$3"
  local request_batch_size="$4"
  local max_num_seqs="$5"
  local gpu_memory_utilization="$6"
  local session="rniah-v2-1-smoke-gpu${gpu}"
  local output_dir="${run_root}/models/${model}/main"
  local log_file="${run_root}/orchestration/${model}.log"
  local revision
  revision="$(revision_for "${model}")"

  if tmux has-session -t "${session}" 2>/dev/null; then
    echo "Session already exists: ${session}" >&2
    return 2
  fi

  mkdir -p "${output_dir}"
  tmux new-session -d -s "${session}" \
    "cd '${repo}' && env \
CUDA_VISIBLE_DEVICES='${gpu}' \
PATH='/home/ubuntu/venvs/realistic-niah-vllm/bin':\"\${PATH}\" \
PYTHONPATH=src \
TOKENIZERS_PARALLELISM=false \
'${python_bin}' scripts/run_realistic_niah.py \
--stimuli '${stimuli}' \
--output-dir '${output_dir}' \
--model '${model}' \
--revision '${revision}' \
--passage-lengths 2000,20000 \
--needle-counts 1,10,30 \
--seeds 1234,1235 \
--prompt-modes '${modes}' \
--query-layout cue_before_query_after \
--cache-dir '${cache}' \
--repo-root '${repo}' \
--tensor-parallel-size 1 \
--max-model-len 32768 \
--gpu-memory-utilization '${gpu_memory_utilization}' \
--max-num-seqs '${max_num_seqs}' \
--request-batch-size '${request_batch_size}' \
--require-clean-git > '${log_file}' 2>&1"
}

mkdir -p "${run_root}/models" "${run_root}/orchestration"
launch 1 Qwen3-1.7B "enumeration_index,enumeration_bullet" 16 16 0.90
launch 2 Qwen3-4B "enumeration_index,enumeration_bullet" 12 12 0.90
launch 3 Qwen3-8B "enumeration_index,enumeration_bullet" 8 8 0.90
launch 4 Qwen3-32B "enumeration_index,enumeration_bullet" 2 1 0.92
launch 5 Gemma4-E4B "enumeration_index,enumeration_bullet" 8 8 0.90
launch 6 Gemma4-12B "direct,enumeration_index,enumeration_bullet" 6 6 0.90
launch 7 GLM-4-9B-0414 "enumeration_index,enumeration_bullet" 8 8 0.90

tmux list-sessions -F '#{session_name}' | grep '^rniah-v2-1-smoke-gpu' | sort
