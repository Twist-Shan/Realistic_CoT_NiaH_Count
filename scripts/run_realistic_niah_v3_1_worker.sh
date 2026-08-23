#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 2 ]]; then
  echo "Usage: $0 RUN_ROOT GPU_ID_OR_WORKER_SLOT" >&2
  exit 2
fi

run_root="$(readlink -f "$1")"
worker_slot="$2"
repo="${REALISTIC_NIAH_REPO_ROOT:-/lambda/nfs/Twist-CoT-Count-Multi-Model-v3/code/Realistic_CoT_NiaH_Count}"
cache="${REALISTIC_NIAH_HF_CACHE:-/lambda/nfs/Twist-CoT-Count-Multi-Model-v3/hf-cache}"
python_bin="${REALISTIC_NIAH_PYTHON:-/home/ubuntu/venvs/realistic-niah-vllm/bin/python}"
device_mode="${REALISTIC_NIAH_DEVICE_MODE:-explicit}"
worker_id="${REALISTIC_NIAH_WORKER_ID:-gpu${worker_slot}}"
stagger_slot="${REALISTIC_NIAH_STAGGER_SLOT:-${worker_slot}}"
claim_grace_seconds="${REALISTIC_NIAH_CLAIM_GRACE_SECONDS:-120}"
model_filter="${REALISTIC_NIAH_MODEL_FILTER:-}"
requested_tensor_parallel_size="${REALISTIC_NIAH_TENSOR_PARALLEL_SIZE:-1}"
stimuli="${run_root}/dataset/stimuli.jsonl"
plan_tsv="${run_root}/orchestration/formal_bundles.tsv"
state_root="${run_root}/orchestration/shard_state"

case "${run_root}" in
  */runs/realistic_niah_v3_1/*) ;;
  *) echo "Refusing unexpected V3.1 run root: ${run_root}" >&2; exit 2 ;;
esac
[[ "${worker_slot}" =~ ^[0-9]+$ ]] \
  || { echo "GPU_ID/worker slot must be non-negative" >&2; exit 2; }
[[ "${worker_id}" =~ ^[A-Za-z0-9_.-]+$ ]] \
  || { echo "REALISTIC_NIAH_WORKER_ID contains unsafe characters" >&2; exit 2; }
[[ "${stagger_slot}" =~ ^[0-9]+$ ]] \
  || { echo "REALISTIC_NIAH_STAGGER_SLOT must be non-negative" >&2; exit 2; }
[[ "${claim_grace_seconds}" =~ ^[0-9]+$ ]] \
  || { echo "REALISTIC_NIAH_CLAIM_GRACE_SECONDS must be non-negative" >&2; exit 2; }
[[ "${requested_tensor_parallel_size}" =~ ^[1-9][0-9]*$ ]] \
  || { echo "REALISTIC_NIAH_TENSOR_PARALLEL_SIZE must be positive" >&2; exit 2; }
case "${device_mode}" in
  explicit)
    nvidia-smi --query-gpu=index --format=csv,noheader \
      | tr -d ' ' | grep -qx "${worker_slot}"
    ;;
  allocated)
    [[ -n "${CUDA_VISIBLE_DEVICES:-}" ]] \
      || { echo "Slurm did not bind a GPU to ${worker_id}" >&2; exit 2; }
    IFS=',' read -r -a allocated_gpus <<< "${CUDA_VISIBLE_DEVICES}"
    if [[ "${#allocated_gpus[@]}" -ne "${requested_tensor_parallel_size}" ]]; then
      echo "Allocated GPU count does not match tensor parallel size for ${worker_id}" >&2
      exit 2
    fi
    nvidia-smi -L >/dev/null
    ;;
  *)
    echo "REALISTIC_NIAH_DEVICE_MODE must be explicit or allocated" >&2
    exit 2
    ;;
esac
test -s "${stimuli}"
test -s "${plan_tsv}"
test -d "${cache}"
test -x "${python_bin}"
test -z "$(git -C "${repo}" status --short)"
mkdir -p "${run_root}/shards" "${state_root}/claims" \
  "${state_root}/completed" "${state_root}/failed" \
  "${state_root}/completed_bundles" "${state_root}/failed_bundles" \
  "${state_root}/failed_attempts" "${state_root}/workers" \
  "${run_root}/orchestration/logs"

engine_settings_for() {
  case "$1" in
    Qwen3-32B) echo "1 1 1 0.92" ;;
    Gemma4-31B) echo "${requested_tensor_parallel_size} 1 1 0.92" ;;
    Gemma4-26B-A4B|Qwen3-14B) echo "1 2 2 0.92" ;;
    Gemma4-12B|Nemotron-Nano-v2-9B|GLM-4-9B-0414|GLM-Z1-9B-0414)
      echo "1 4 4 0.90" ;;
    Qwen3-8B|Gemma4-E4B|Ministral-3-Instruct-8B|Ministral-3-Reasoning-8B)
      echo "1 6 6 0.90" ;;
    Qwen3-4B|Nemotron-3-Nano-4B) echo "1 8 8 0.90" ;;
    *) echo "No V3.1 engine settings for $1" >&2; return 2 ;;
  esac
}

write_two_row_marker() {
  local destination="$1"
  local header="$2"
  local row="$3"
  local temporary
  temporary="$(mktemp "${destination}.tmp.XXXXXX")"
  printf '%s\n%s\n' "${header}" "${row}" > "${temporary}"
  mv -f -- "${temporary}" "${destination}"
}

archive_invalid_bundle_marker() {
  local bundle_id="$1"
  local marker="${state_root}/completed_bundles/${bundle_id}.tsv"
  local archive_root="${state_root}/failed_attempts/${bundle_id}"
  local stamp
  [[ -e "${marker}" ]] || return 0
  stamp="$(date -u +%Y%m%dT%H%M%SZ).${worker_id}.$RANDOM"
  mkdir -p "${archive_root}"
  mv "${marker}" "${archive_root}/invalid-completion.${stamp}.tsv" \
    2>/dev/null || return 0
}

archive_previous_attempt_if_safe() {
  local bundle_id="$1"
  local claim_dir="${state_root}/claims/${bundle_id}"
  local failed_file="${state_root}/failed_bundles/${bundle_id}.tsv"
  local archive_root="${state_root}/failed_attempts/${bundle_id}"
  local prior_host=""
  local prior_pid=""
  local prior_scheduler_job_id=""
  local claim_age_seconds=0
  local claim_mtime=0
  local now_epoch=0
  local stamp

  if [[ ! -d "${claim_dir}" ]]; then
    if [[ -s "${failed_file}" ]]; then
      stamp="$(date -u +%Y%m%dT%H%M%SZ).${worker_id}.$RANDOM"
      mkdir -p "${archive_root}"
      mv "${failed_file}" "${archive_root}/failed.${stamp}.tsv" 2>/dev/null \
        || return 1
    fi
    return 0
  fi
  if [[ -s "${claim_dir}/claim.tsv" ]] \
    && [[ "$(wc -l < "${claim_dir}/claim.tsv")" -eq 2 ]]; then
    prior_host="$(awk -F $'\t' 'NR==2 {print $4}' "${claim_dir}/claim.tsv")"
    prior_pid="$(awk -F $'\t' 'NR==2 {print $3}' "${claim_dir}/claim.tsv")"
    prior_scheduler_job_id="$(awk -F $'\t' 'NR==2 {print $8}' "${claim_dir}/claim.tsv")"
  fi
  if [[ -n "${prior_scheduler_job_id}" ]] \
    && command -v squeue >/dev/null 2>&1 \
    && [[ -n "$(squeue -h -j "${prior_scheduler_job_id}" 2>/dev/null)" ]]; then
    return 1
  fi
  if [[ -z "${prior_scheduler_job_id}" ]] \
    && [[ "${prior_host}" == "$(hostname)" ]] \
    && [[ "${prior_pid}" =~ ^[0-9]+$ ]] \
    && kill -0 "${prior_pid}" 2>/dev/null; then
    return 1
  fi
  if [[ -z "${prior_host}" ]]; then
    claim_mtime="$(stat -c %Y "${claim_dir}")"
    now_epoch="$(date +%s)"
    claim_age_seconds="$((now_epoch - claim_mtime))"
    if [[ "${claim_age_seconds}" -lt "${claim_grace_seconds}" ]]; then
      return 1
    fi
  fi
  if [[ "${prior_host}" != "$(hostname)" ]] \
    && [[ -z "${prior_scheduler_job_id}" ]] \
    && [[ ! -s "${failed_file}" ]]; then
    # A legacy claim on another host cannot be proven stale safely.
    return 1
  fi
  stamp="$(date -u +%Y%m%dT%H%M%SZ).${worker_id}.$RANDOM"
  mkdir -p "${archive_root}"
  mv "${claim_dir}" "${archive_root}/claim.${stamp}" 2>/dev/null \
    || return 1
  if [[ -s "${failed_file}" ]]; then
    mv "${failed_file}" "${archive_root}/failed.${stamp}.tsv"
  fi
  return 0
}

worker_file="${state_root}/workers/${worker_id}.tsv"
printf "worker_id\tworker_slot\tpid\thostname\tstarted_at_utc\tstatus\n" > "${worker_file}"
printf "%s\t%s\t%s\t%s\t%s\trunning\n" "${worker_id}" "${worker_slot}" "$$" "$(hostname)" \
  "$(date -u +%Y-%m-%dT%H:%M:%SZ)" >> "${worker_file}"
sleep "$((stagger_slot * 5))"

while IFS=$'\t' read -r \
  bundle_id priority model expected_logical_shards expected_requests revision \
  prompt_modes logical_task_ids
do
  [[ "${bundle_id}" == "bundle_id" ]] && continue
  [[ -z "${model_filter}" || "${model}" == "${model_filter}" ]] || continue
  bundle_completed_file="${state_root}/completed_bundles/${bundle_id}.tsv"
  failed_file="${state_root}/failed_bundles/${bundle_id}.tsv"
  claim_dir="${state_root}/claims/${bundle_id}"
  if PYTHONPATH="${repo}/src" "${python_bin}" \
    "${repo}/scripts/audit_realistic_niah_v3_1_shard_state.py" \
      --run-root "${run_root}" --bundle-id "${bundle_id}" --quiet \
      >/dev/null 2>&1; then
    continue
  fi
  archive_invalid_bundle_marker "${bundle_id}"
  archive_previous_attempt_if_safe "${bundle_id}" || continue
  attempt_id="$(date -u +%Y%m%dT%H%M%SZ).${worker_id}.$RANDOM"
  claim_temporary="$(mktemp "${state_root}/claims/.${bundle_id}.${worker_id}.XXXXXX")"
  printf "bundle_id\tworker_slot\tpid\thostname\tclaimed_at_utc\tattempt_id\tworker_id\tscheduler_job_id\n" \
    > "${claim_temporary}"
  printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" \
    "${bundle_id}" "${worker_slot}" "$$" \
    "$(hostname)" "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "${attempt_id}" \
    "${worker_id}" "${SLURM_JOB_ID:-}" \
    >> "${claim_temporary}"
  if ! mkdir "${claim_dir}" 2>/dev/null; then
    rm -f -- "${claim_temporary}"
    continue
  fi
  mv -- "${claim_temporary}" "${claim_dir}/claim.tsv"
  read -r tensor_parallel_size request_batch_size max_num_seqs gpu_utilization \
    < <(engine_settings_for "${model}")
  log_file="${run_root}/orchestration/logs/${bundle_id}.${attempt_id}.log"
  inference_environment=(
    env
    "PATH=$(dirname "${python_bin}"):${PATH}"
    "PYTHONPATH=src"
    "TOKENIZERS_PARALLELISM=false"
  )
  if [[ "${device_mode}" == "explicit" ]]; then
    inference_environment+=("CUDA_VISIBLE_DEVICES=${worker_slot}")
  fi
  if (
    cd "${repo}"
    "${inference_environment[@]}" \
      "${python_bin}" scripts/run_realistic_niah_v3_1_model_bundle.py \
        --stimuli "${stimuli}" --run-root "${run_root}" \
        --model "${model}" --revision "${revision}" \
        --query-layout cue_before_query_after \
        --cache-dir "${cache}" --repo-root "${repo}" \
        --tensor-parallel-size "${tensor_parallel_size}" --max-model-len 32768 \
        --gpu-memory-utilization "${gpu_utilization}" \
        --max-num-seqs "${max_num_seqs}" \
        --request-batch-size "${request_batch_size}" --require-clean-git
  ) > "${log_file}" 2>&1; then
    IFS=',' read -r -a task_ids <<< "${logical_task_ids}"
    [[ "${#task_ids[@]}" -eq "${expected_logical_shards}" ]]
    for task_id in "${task_ids[@]}"; do
      output_dir="${run_root}/shards/${task_id}/main"
      "${python_bin}" -c \
        'import json,sys; p=json.load(open(sys.argv[1])); e=int(sys.argv[2]); assert p["protocol_version"]=="realistic_niah_v3_1"; assert p["completed_requests"]==e==p["expected_requests"]; assert p["prompt_payload_storage"]=="sha256_only"' \
        "${output_dir}/run_manifest.json" 3360
      prompt_mode="${task_id##*__}"
      completed_file="${state_root}/completed/${task_id}.tsv"
      printf -v marker_row "%s\t%s\t%s\t%s\t%s\t%s" "${task_id}" "${model}" \
        "${prompt_mode}" "${worker_id}" "${attempt_id}" \
        "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
      write_two_row_marker "${completed_file}" \
        $'task_id\tmodel\tprompt_mode\tworker_id\tattempt_id\tcompleted_at_utc' \
        "${marker_row}"
    done
    printf -v marker_row "%s\t%s\t%s\t%s\t%s\t%s\t%s" "${bundle_id}" "${model}" \
      "${expected_logical_shards}" "${expected_requests}" "${worker_id}" \
      "${attempt_id}" "$(date -u +%Y-%m-%dT%H:%M:%SZ)" \
    write_two_row_marker "${bundle_completed_file}" \
      $'bundle_id\tmodel\tlogical_shards\trequests\tworker_id\tattempt_id\tcompleted_at_utc' \
      "${marker_row}"
  else
    exit_code=$?
    printf -v marker_row "%s\t%s\t%s\t%s\t%s\t%s\t%s" "${bundle_id}" "${model}" \
      "${prompt_modes}" "${worker_id}" "${attempt_id}" "${exit_code}" \
      "${log_file}"
    write_two_row_marker "${failed_file}" \
      $'bundle_id\tmodel\tprompt_modes\tworker_id\tattempt_id\texit_code\tlog' \
      "${marker_row}"
    exit "${exit_code}"
  fi
done < "${plan_tsv}"

printf "%s\t%s\t%s\t%s\t%s\tcompleted\n" "${worker_id}" "${worker_slot}" "$$" "$(hostname)" \
  "$(date -u +%Y-%m-%dT%H:%M:%SZ)" >> "${worker_file}"
