#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 3 ]]; then
  echo "Usage: $0 RUN_ROOT GPU_IDS_CSV TASK_ID" >&2
  exit 2
fi

run_root="$(readlink -f "$1")"
gpu_ids="$2"
requested_task_id="$3"
repo="${REALISTIC_NIAH_REPO_ROOT:-/lambda/nfs/Twist-CoT-Count-Multi-Model-v3/code/Realistic_CoT_NiaH_Count}"
cache="${REALISTIC_NIAH_HF_CACHE:-/lambda/nfs/Twist-CoT-Count-Multi-Model-v3/hf-cache}"
python_bin="${REALISTIC_NIAH_PYTHON:-/home/ubuntu/venvs/realistic-niah-vllm/bin/python}"
stimuli="${run_root}/dataset/stimuli.jsonl"
plan_tsv="${run_root}/orchestration/formal_shards.tsv"
state_root="${run_root}/orchestration/shard_state"

case "${run_root}" in
  */runs/realistic_niah_v3/*) ;;
  *)
    echo "Refusing unexpected V3 run root: ${run_root}" >&2
    exit 2
    ;;
esac
if ! [[ "${gpu_ids}" =~ ^[0-9]+(,[0-9]+)*$ ]]; then
  echo "GPU_IDS_CSV must contain comma-separated non-negative integers" >&2
  exit 2
fi
if ! [[ "${requested_task_id}" =~ ^[A-Za-z0-9_.-]+$ ]]; then
  echo "Invalid V3 task id: ${requested_task_id}" >&2
  exit 2
fi

IFS=',' read -r -a allocated_gpus <<< "${gpu_ids}"
if [[ "$(printf '%s\n' "${allocated_gpus[@]}" | sort -u | wc -l)" \
  -ne "${#allocated_gpus[@]}" ]]; then
  echo "GPU allocation contains duplicates: ${gpu_ids}" >&2
  exit 2
fi
visible_gpus="$(nvidia-smi --query-gpu=index --format=csv,noheader | tr -d ' ')"
for gpu_id in "${allocated_gpus[@]}"; do
  if ! grep -qx "${gpu_id}" <<< "${visible_gpus}"; then
    echo "GPU ${gpu_id} is not visible" >&2
    exit 2
  fi
done

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

task_line="$(
  awk -F $'\t' -v task_id="${requested_task_id}" \
    '$1 == task_id {print; found=1} END {if (!found) exit 1}' \
    "${plan_tsv}"
)" || {
  echo "Task not found in frozen plan: ${requested_task_id}" >&2
  exit 2
}
IFS=$'\t' read -r \
  task_id priority model prompt_mode output_collection expected_requests \
  revision gpus_required tensor_parallel_size request_batch_size \
  max_num_seqs gpu_utilization max_model_len <<< "${task_line}"

if [[ "${task_id}" != "${requested_task_id}" ]]; then
  echo "Resolved task id does not match request" >&2
  exit 2
fi
if [[ "${#allocated_gpus[@]}" -ne "${gpus_required}" ]]; then
  echo "Task ${task_id} requires ${gpus_required} GPU(s), got ${gpu_ids}" >&2
  exit 2
fi
if [[ "${tensor_parallel_size}" -ne "${gpus_required}" ]]; then
  echo "Task ${task_id} has inconsistent TP/GPU resource fields" >&2
  exit 2
fi

archive_previous_attempt_if_safe() {
  local claim_dir="${state_root}/claims/${task_id}"
  local failed_file="${state_root}/failed/${task_id}.tsv"
  local archive_root="${state_root}/failed_attempts/${task_id}"
  local prior_host=""
  local prior_pid=""
  local stamp

  if [[ ! -d "${claim_dir}" ]]; then
    if [[ -s "${failed_file}" ]]; then
      stamp="$(date -u +%Y%m%dT%H%M%SZ).gpu${allocated_gpus[0]}.$RANDOM"
      mkdir -p "${archive_root}"
      mv "${failed_file}" "${archive_root}/failed.${stamp}.tsv"
    fi
    return 0
  fi
  if [[ -s "${claim_dir}/claim.tsv" ]]; then
    prior_pid="$(awk -F $'\t' 'NR==2 {print $3}' "${claim_dir}/claim.tsv")"
    prior_host="$(awk -F $'\t' 'NR==2 {print $4}' "${claim_dir}/claim.tsv")"
  fi
  if [[ "${prior_host}" == "$(hostname)" ]] \
    && [[ "${prior_pid}" =~ ^[0-9]+$ ]] \
    && kill -0 "${prior_pid}" 2>/dev/null; then
    echo "Task ${task_id} is already claimed by live PID ${prior_pid}" >&2
    return 1
  fi
  if [[ ! -s "${failed_file}" && -z "${prior_host}" ]]; then
    echo "Task ${task_id} has an unverified existing claim" >&2
    return 1
  fi
  stamp="$(date -u +%Y%m%dT%H%M%SZ).gpu${allocated_gpus[0]}.$RANDOM"
  mkdir -p "${archive_root}"
  mv "${claim_dir}" "${archive_root}/claim.${stamp}"
  if [[ -s "${failed_file}" ]]; then
    mv "${failed_file}" "${archive_root}/failed.${stamp}.tsv"
  fi
}

completed_file="${state_root}/completed/${task_id}.tsv"
claim_dir="${state_root}/claims/${task_id}"
failed_file="${state_root}/failed/${task_id}.tsv"
if [[ -s "${completed_file}" ]]; then
  echo "Task ${task_id} is already complete; refusing to rerun it." >&2
  exit 2
fi
archive_previous_attempt_if_safe
if ! mkdir "${claim_dir}" 2>/dev/null; then
  echo "Could not claim task ${task_id}" >&2
  exit 2
fi

gpu_tag="${gpu_ids//,/-}"
attempt_id="$(date -u +%Y%m%dT%H%M%SZ).gpu${gpu_tag}.$RANDOM"
claimed_at="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
printf "task_id\tgpu_ids\tpid\thostname\tclaimed_at_utc\tattempt_id\n" \
  > "${claim_dir}/claim.tsv"
printf "%s\t%s\t%s\t%s\t%s\t%s\n" \
  "${task_id}" "${gpu_ids}" "$$" "$(hostname)" "${claimed_at}" \
  "${attempt_id}" >> "${claim_dir}/claim.tsv"

worker_file="${state_root}/workers/${task_id}.${attempt_id}.tsv"
printf "task_id\tgpu_ids\tpid\thostname\tstarted_at_utc\tstatus\n" \
  > "${worker_file}"
printf "%s\t%s\t%s\t%s\t%s\trunning\n" \
  "${task_id}" "${gpu_ids}" "$$" "$(hostname)" "${claimed_at}" \
  >> "${worker_file}"

output_dir="${run_root}/shards/${task_id}/main"
log_file="${run_root}/orchestration/logs/${task_id}.${attempt_id}.log"
mkdir -p "${output_dir}"

if (
  cd "${repo}"
  env \
    CUDA_VISIBLE_DEVICES="${gpu_ids}" \
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
      --tensor-parallel-size "${tensor_parallel_size}" \
      --max-model-len "${max_model_len}" \
      --gpu-memory-utilization "${gpu_utilization}" \
      --max-num-seqs "${max_num_seqs}" \
      --request-batch-size "${request_batch_size}" \
      --require-clean-git
) > "${log_file}" 2>&1 \
  && "${python_bin}" -c \
    'import json,sys; p=json.load(open(sys.argv[1],encoding="utf-8")); e=int(sys.argv[2]); assert p["protocol_version"] == "realistic_niah_v3"; assert p["completed_requests"] == e == p["expected_requests"]' \
    "${output_dir}/run_manifest.json" "${expected_requests}" \
    >> "${log_file}" 2>&1; then
  printf "task_id\tmodel\tprompt_mode\tgpu_ids\tattempt_id\tcompleted_at_utc\n" \
    > "${completed_file}"
  printf "%s\t%s\t%s\t%s\t%s\t%s\n" \
    "${task_id}" "${model}" "${prompt_mode}" "${gpu_ids}" \
    "${attempt_id}" "$(date -u +%Y-%m-%dT%H:%M:%SZ)" \
    >> "${completed_file}"
  printf "%s\t%s\t%s\t%s\t%s\tcompleted\n" \
    "${task_id}" "${gpu_ids}" "$$" "$(hostname)" \
    "$(date -u +%Y-%m-%dT%H:%M:%SZ)" >> "${worker_file}"
else
  exit_code=$?
  printf "task_id\tmodel\tprompt_mode\tgpu_ids\tattempt_id\texit_code\tfailed_at_utc\tlog\n" \
    > "${failed_file}"
  printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" \
    "${task_id}" "${model}" "${prompt_mode}" "${gpu_ids}" \
    "${attempt_id}" "${exit_code}" "$(date -u +%Y-%m-%dT%H:%M:%SZ)" \
    "${log_file}" >> "${failed_file}"
  printf "%s\t%s\t%s\t%s\t%s\tfailed\n" \
    "${task_id}" "${gpu_ids}" "$$" "$(hostname)" \
    "$(date -u +%Y-%m-%dT%H:%M:%SZ)" >> "${worker_file}"
  exit "${exit_code}"
fi
