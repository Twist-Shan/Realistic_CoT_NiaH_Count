#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 2 ]]; then
  echo "Usage: $0 RUN_ROOT GPU_ID" >&2
  exit 2
fi

run_root="$(readlink -f "$1")"
gpu_id="$2"
repo="${REALISTIC_NIAH_REPO_ROOT:-/lambda/nfs/Twist-CoT-Count-Multi-Model-v2/code/Realistic_CoT_NiaH_Count}"
cache="${REALISTIC_NIAH_HF_CACHE:-/lambda/nfs/Twist-CoT-Count-Multi-Model-v2/hf-cache}"
python_bin="${REALISTIC_NIAH_PYTHON:-/home/ubuntu/venvs/realistic-niah-vllm/bin/python}"
stimuli="${run_root}/dataset/stimuli.jsonl"
plan_tsv="${run_root}/orchestration/formal_shards.tsv"
state_root="${run_root}/orchestration/shard_state"

case "${run_root}" in
  /lambda/nfs/Twist-CoT-Count-Multi-Model-v2/runs/realistic_niah_v2/eight_models_formal_*)
    ;;
  *)
    echo "Refusing unexpected run root: ${run_root}" >&2
    exit 2
    ;;
esac

if ! [[ "${gpu_id}" =~ ^[0-7]$ ]]; then
  echo "GPU_ID must be an integer from 0 through 7" >&2
  exit 2
fi

test -s "${stimuli}"
test -s "${plan_tsv}"
test -d "${cache}"
test -x "${python_bin}"
test -z "$(git -C "${repo}" status --short)"
mkdir -p \
  "${run_root}/shards" \
  "${state_root}/claims" \
  "${state_root}/completed" \
  "${state_root}/failed" \
  "${state_root}/workers"

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

worker_file="${state_root}/workers/gpu${gpu_id}.tsv"
printf "gpu_id\tpid\thostname\tstarted_at_utc\tstatus\n" > "${worker_file}"
printf "%s\t%s\t%s\t%s\trunning\n" \
  "${gpu_id}" "$$" "$(hostname)" "$(date -u +%Y-%m-%dT%H:%M:%SZ)" \
  >> "${worker_file}"

# Avoid eight simultaneous Python/vLLM startups hammering the shared cache.
sleep "$((gpu_id * 5))"

while IFS=$'\t' read -r \
  task_id priority model prompt_mode output_collection expected_requests revision
do
  if [[ "${task_id}" == "task_id" ]]; then
    continue
  fi
  completed_file="${state_root}/completed/${task_id}.tsv"
  claim_dir="${state_root}/claims/${task_id}"
  if [[ -s "${completed_file}" ]]; then
    continue
  fi
  if ! mkdir "${claim_dir}" 2>/dev/null; then
    continue
  fi

  claimed_at="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  printf "task_id\tgpu_id\tpid\thostname\tclaimed_at_utc\n" \
    > "${claim_dir}/claim.tsv"
  printf "%s\t%s\t%s\t%s\t%s\n" \
    "${task_id}" "${gpu_id}" "$$" "$(hostname)" "${claimed_at}" \
    >> "${claim_dir}/claim.tsv"

  read -r request_batch_size max_num_seqs < <(engine_limits_for "${model}")
  output_dir="${run_root}/shards/${task_id}/main"
  log_file="${run_root}/orchestration/shard.${task_id}.log"
  mkdir -p "${output_dir}"

  if (
    cd "${repo}"
    env \
      CUDA_VISIBLE_DEVICES="${gpu_id}" \
      PATH="/home/ubuntu/venvs/realistic-niah-vllm/bin:${PATH}" \
      PYTHONPATH=src \
      TOKENIZERS_PARALLELISM=false \
      "${python_bin}" scripts/run_realistic_niah.py \
        --stimuli "${stimuli}" \
        --output-dir "${output_dir}" \
        --model "${model}" \
        --revision "${revision}" \
        --prompt-modes "${prompt_mode}" \
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
    "${python_bin}" -c \
      'import json,sys; p=json.load(open(sys.argv[1],encoding="utf-8")); e=int(sys.argv[2]); assert p["completed_requests"] == e == p["expected_requests"]' \
      "${output_dir}/run_manifest.json" "${expected_requests}"
    printf "task_id\tmodel\tprompt_mode\tgpu_id\tcompleted_at_utc\n" \
      > "${completed_file}"
    printf "%s\t%s\t%s\t%s\t%s\n" \
      "${task_id}" "${model}" "${prompt_mode}" "${gpu_id}" \
      "$(date -u +%Y-%m-%dT%H:%M:%SZ)" >> "${completed_file}"
  else
    exit_code=$?
    failed_file="${state_root}/failed/${task_id}.tsv"
    printf "task_id\tmodel\tprompt_mode\tgpu_id\texit_code\tfailed_at_utc\n" \
      > "${failed_file}"
    printf "%s\t%s\t%s\t%s\t%s\t%s\n" \
      "${task_id}" "${model}" "${prompt_mode}" "${gpu_id}" \
      "${exit_code}" "$(date -u +%Y-%m-%dT%H:%M:%SZ)" \
      >> "${failed_file}"
    printf "%s\t%s\t%s\t%s\tfailed(%s)\n" \
      "${gpu_id}" "$$" "$(hostname)" \
      "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "${exit_code}" \
      >> "${worker_file}"
    exit "${exit_code}"
  fi
done < "${plan_tsv}"

printf "%s\t%s\t%s\t%s\tcompleted\n" \
  "${gpu_id}" "$$" "$(hostname)" "$(date -u +%Y-%m-%dT%H:%M:%SZ)" \
  >> "${worker_file}"
