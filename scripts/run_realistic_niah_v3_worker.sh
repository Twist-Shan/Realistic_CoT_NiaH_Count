#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 2 ]]; then
  echo "Usage: $0 RUN_ROOT GPU_ID" >&2
  exit 2
fi

run_root="$(readlink -f "$1")"
gpu_id="$2"
repo="${REALISTIC_NIAH_REPO_ROOT:-/lambda/nfs/Twist-CoT-Count-Multi-Model-v3/code/Realistic_CoT_NiaH_Count}"
cache="${REALISTIC_NIAH_HF_CACHE:-/lambda/nfs/Twist-CoT-Count-Multi-Model-v3/hf-cache}"
python_bin="${REALISTIC_NIAH_PYTHON:-/home/ubuntu/venvs/realistic-niah-vllm/bin/python}"
stimuli="${run_root}/dataset/stimuli.jsonl"
plan_tsv="${run_root}/orchestration/formal_shards.tsv"
state_root="${run_root}/orchestration/shard_state"

case "${run_root}" in
  */runs/realistic_niah_v3/*)
    ;;
  *)
    echo "Refusing unexpected V3 run root: ${run_root}" >&2
    exit 2
    ;;
esac
if ! [[ "${gpu_id}" =~ ^[0-9]+$ ]]; then
  echo "GPU_ID must be a non-negative integer" >&2
  exit 2
fi
if ! nvidia-smi --query-gpu=index --format=csv,noheader \
  | tr -d ' ' | grep -qx "${gpu_id}"; then
  echo "GPU ${gpu_id} is not visible" >&2
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
  "${state_root}/failed_attempts" \
  "${state_root}/workers" \
  "${run_root}/orchestration/logs"

engine_settings_for() {
  case "$1" in
    Qwen3-32B|Gemma4-31B) echo "1 1 0.92" ;;
    Gemma4-26B-A4B|Qwen3-14B) echo "2 2 0.92" ;;
    Gemma4-12B|Nemotron-Nano-v2-9B|GLM-4-9B-0414|GLM-Z1-9B-0414)
      echo "4 4 0.90"
      ;;
    Qwen3-8B|Gemma4-E4B|Ministral-3-Instruct-8B|Ministral-3-Reasoning-8B)
      echo "6 6 0.90"
      ;;
    Qwen3-4B|Nemotron-3-Nano-4B) echo "8 8 0.90" ;;
    *)
      echo "No V3 engine settings registered for $1" >&2
      return 2
      ;;
  esac
}

archive_previous_attempt_if_safe() {
  local task_id="$1"
  local claim_dir="${state_root}/claims/${task_id}"
  local failed_file="${state_root}/failed/${task_id}.tsv"
  local archive_root="${state_root}/failed_attempts/${task_id}"
  local prior_host=""
  local prior_pid=""
  local stamp

  if [[ ! -d "${claim_dir}" ]]; then
    if [[ -s "${failed_file}" ]]; then
      stamp="$(date -u +%Y%m%dT%H%M%SZ).gpu${gpu_id}.$RANDOM"
      mkdir -p "${archive_root}"
      mv "${failed_file}" "${archive_root}/failed.${stamp}.tsv"
    fi
    return 0
  fi
  if [[ -s "${claim_dir}/claim.tsv" ]]; then
    prior_host="$(awk -F $'\t' 'NR==2 {print $4}' "${claim_dir}/claim.tsv")"
    prior_pid="$(awk -F $'\t' 'NR==2 {print $3}' "${claim_dir}/claim.tsv")"
  fi
  if [[ "${prior_host}" == "$(hostname)" ]] \
    && [[ "${prior_pid}" =~ ^[0-9]+$ ]] \
    && kill -0 "${prior_pid}" 2>/dev/null; then
    return 1
  fi
  if [[ ! -s "${failed_file}" && -z "${prior_host}" ]]; then
    return 1
  fi
  stamp="$(date -u +%Y%m%dT%H%M%SZ).gpu${gpu_id}.$RANDOM"
  mkdir -p "${archive_root}"
  if ! mv "${claim_dir}" "${archive_root}/claim.${stamp}" 2>/dev/null; then
    return 1
  fi
  if [[ -s "${failed_file}" ]]; then
    mv "${failed_file}" "${archive_root}/failed.${stamp}.tsv"
  fi
  return 0
}

worker_file="${state_root}/workers/gpu${gpu_id}.tsv"
printf "gpu_id\tpid\thostname\tstarted_at_utc\tstatus\n" > "${worker_file}"
printf "%s\t%s\t%s\t%s\trunning\n" \
  "${gpu_id}" "$$" "$(hostname)" "$(date -u +%Y-%m-%dT%H:%M:%SZ)" \
  >> "${worker_file}"
sleep "$((gpu_id * 5))"

while IFS=$'\t' read -r \
  task_id priority model prompt_mode output_collection expected_requests revision
do
  if [[ "${task_id}" == "task_id" ]]; then
    continue
  fi
  completed_file="${state_root}/completed/${task_id}.tsv"
  claim_dir="${state_root}/claims/${task_id}"
  failed_file="${state_root}/failed/${task_id}.tsv"
  if [[ -s "${completed_file}" ]]; then
    continue
  fi
  archive_previous_attempt_if_safe "${task_id}" || continue
  if ! mkdir "${claim_dir}" 2>/dev/null; then
    continue
  fi

  attempt_id="$(date -u +%Y%m%dT%H%M%SZ).gpu${gpu_id}.$RANDOM"
  claimed_at="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  printf "task_id\tgpu_id\tpid\thostname\tclaimed_at_utc\tattempt_id\n" \
    > "${claim_dir}/claim.tsv"
  printf "%s\t%s\t%s\t%s\t%s\t%s\n" \
    "${task_id}" "${gpu_id}" "$$" "$(hostname)" "${claimed_at}" \
    "${attempt_id}" >> "${claim_dir}/claim.tsv"

  read -r request_batch_size max_num_seqs gpu_utilization \
    < <(engine_settings_for "${model}")
  output_dir="${run_root}/shards/${task_id}/main"
  log_file="${run_root}/orchestration/logs/${task_id}.${attempt_id}.log"
  mkdir -p "${output_dir}"

  if (
    cd "${repo}"
    env \
      CUDA_VISIBLE_DEVICES="${gpu_id}" \
      PATH="$(dirname "${python_bin}"):${PATH}" \
      PYTHONPATH=src \
      TOKENIZERS_PARALLELISM=false \
      "${python_bin}" scripts/run_realistic_niah_v3.py \
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
        --gpu-memory-utilization "${gpu_utilization}" \
        --max-num-seqs "${max_num_seqs}" \
        --request-batch-size "${request_batch_size}" \
        --require-clean-git
  ) > "${log_file}" 2>&1; then
    "${python_bin}" -c \
      'import json,sys; p=json.load(open(sys.argv[1],encoding="utf-8")); e=int(sys.argv[2]); assert p["protocol_version"] == "realistic_niah_v3"; assert p["completed_requests"] == e == p["expected_requests"]' \
      "${output_dir}/run_manifest.json" "${expected_requests}"
    printf "task_id\tmodel\tprompt_mode\tgpu_id\tattempt_id\tcompleted_at_utc\n" \
      > "${completed_file}"
    printf "%s\t%s\t%s\t%s\t%s\t%s\n" \
      "${task_id}" "${model}" "${prompt_mode}" "${gpu_id}" \
      "${attempt_id}" "$(date -u +%Y-%m-%dT%H:%M:%SZ)" \
      >> "${completed_file}"
  else
    exit_code=$?
    printf "task_id\tmodel\tprompt_mode\tgpu_id\tattempt_id\texit_code\tfailed_at_utc\tlog\n" \
      > "${failed_file}"
    printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" \
      "${task_id}" "${model}" "${prompt_mode}" "${gpu_id}" \
      "${attempt_id}" "${exit_code}" "$(date -u +%Y-%m-%dT%H:%M:%SZ)" \
      "${log_file}" >> "${failed_file}"
    exit "${exit_code}"
  fi
done < "${plan_tsv}"

printf "%s\t%s\t%s\t%s\tcompleted\n" \
  "${gpu_id}" "$$" "$(hostname)" "$(date -u +%Y-%m-%dT%H:%M:%SZ)" \
  >> "${worker_file}"
