#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 8 ]]; then
  echo "Usage: $0 GROUP_ID TIER MIN_MB MAX_MB WORKER0 WORKER1 WORKER2 WORKER3" >&2
  exit 2
fi

group_id="$1"
tier="$2"
min_memory_mb="$3"
max_memory_mb="$4"
shift 4
worker_specs=("$@")

expected_commit="0afd1adcf2e4b9033805323b9e396220c29afb9e"
work_root="${_CONDOR_SCRATCH_DIR:-$PWD}"
repo="${work_root}/repo"
run_root="${repo}/runs/realistic_niah_v3_1/chtc_a100_${CLUSTER:-manual}_${PROCESS:-0}_${group_id}"
cache="${work_root}/hf-cache"
overlay="${work_root}/python-overlay"
partial_meta="${run_root}/partial_meta/${group_id}"
result_archive="${PWD}/chtc_v31_part_${group_id}.tar.gz"

archive_results() {
  status=$?
  set +e
  mkdir -p "${partial_meta}"
  printf '{"group_id":"%s","tier":"%s","exit_status":%s,"cluster":"%s","process":"%s","completed_at_utc":"%s"}\n' \
    "${group_id}" "${tier}" "${status}" "${CLUSTER:-unknown}" \
    "${PROCESS:-unknown}" "$(date -u +%Y-%m-%dT%H:%M:%SZ)" \
    > "${partial_meta}/exit.json"
  if [[ -d "${run_root}/shards" ]]; then
    tar -czf "${result_archive}" -C "${run_root}" shards partial_meta
  else
    tar -czf "${result_archive}" -C "${run_root}" partial_meta
  fi
  exit "${status}"
}
trap archive_results EXIT

[[ "${group_id}" =~ ^a100_(40|80)_[0-9]+$ ]]
[[ "${tier}" == "40gb" || "${tier}" == "80gb" ]]
[[ "${min_memory_mb}" =~ ^[0-9]+$ && "${max_memory_mb}" =~ ^[0-9]+$ ]]
(( min_memory_mb < max_memory_mb ))

IFS=',' read -r -a allocated_gpus \
  <<< "${CUDA_VISIBLE_DEVICES:?HTCondor did not assign CUDA devices}"
if [[ "${#allocated_gpus[@]}" -ne 4 ]]; then
  echo "Expected exactly four assigned GPUs, got ${#allocated_gpus[@]}" >&2
  exit 2
fi

mkdir -p "${partial_meta}/logs"
nvidia-smi --query-gpu=index,name,uuid,memory.total,driver_version \
  --format=csv,noheader > "${partial_meta}/nvidia_smi.csv"
if [[ "$(wc -l < "${partial_meta}/nvidia_smi.csv")" -ne 4 ]]; then
  echo "nvidia-smi did not expose exactly four GPUs" >&2
  exit 2
fi
if grep -Eiv 'A100' "${partial_meta}/nvidia_smi.csv" >/dev/null; then
  echo "The allocation contains a non-A100 GPU" >&2
  exit 2
fi
if grep -Ei 'MIG' "${partial_meta}/nvidia_smi.csv" >/dev/null; then
  echo "MIG devices are excluded from this formal run" >&2
  exit 2
fi
while IFS=',' read -r _index _name _uuid memory_with_unit _driver; do
  memory_mb="$(tr -dc '0-9' <<< "${memory_with_unit}")"
  if (( memory_mb < min_memory_mb || memory_mb >= max_memory_mb )); then
    echo "GPU memory ${memory_mb}MB falls outside [${min_memory_mb}, ${max_memory_mb})" >&2
    exit 2
  fi
done < "${partial_meta}/nvidia_smi.csv"

git clone -b codex/realistic-consolidation realistic_v31.bundle "${repo}"
[[ "$(git -C "${repo}" rev-parse HEAD)" == "${expected_commit}" ]]
mkdir -p "${run_root}" "${cache}" "${overlay}"
tar -xzf realistic_v31_dataset.tar.gz -C "${run_root}"

required_free_kb=200000000
[[ "${tier}" == "80gb" ]] && required_free_kb=450000000
available_kb="$(df -Pk "${cache}" | awk 'NR==2 {print $4}')"
if [[ ! "${available_kb}" =~ ^[0-9]+$ ]] || (( available_kb < required_free_kb )); then
  echo "Insufficient cache filesystem space for ${tier}: ${available_kb}KB" >&2
  df -hT "${cache}" >&2 || true
  exit 2
fi

if ! python3 - <<'PY'
import transformers, vllm
assert transformers.__version__ == "5.14.1", transformers.__version__
assert vllm.__version__ == "0.25.1", vllm.__version__
PY
then
  python3 -m pip install --disable-pip-version-check --no-cache-dir \
    --target "${overlay}" "transformers==5.14.1" "mistral-common>=1.8.6,<2"
fi

export PYTHONPATH="${overlay}:${repo}/src"
export HF_HOME="${cache}"
export TOKENIZERS_PARALLELISM=false
python3 "${repo}/scripts/prepare_realistic_niah_v3_1.py" \
  --run-root "${run_root}" --repo-root "${repo}" \
  > "${partial_meta}/prepare_stdout.json"
cp "${run_root}/orchestration/prepare_audit.json" "${partial_meta}/prepare_audit.json"
python3 -m pip freeze > "${partial_meta}/pip_freeze.txt"
df -hT > "${partial_meta}/filesystems.txt"

engine_settings_for() {
  case "$1" in
    Qwen3-32B|Gemma4-31B) echo "1 1 0.92" ;;
    Gemma4-26B-A4B|Qwen3-14B) echo "2 2 0.92" ;;
    Gemma4-12B|Nemotron-Nano-v2-9B) echo "4 4 0.90" ;;
    Qwen3-4B) echo "4 4 0.88" ;;
    Qwen3-8B|Gemma4-E4B|Nemotron-3-Nano-4B|GLM-4-9B-0414|GLM-Z1-9B-0414|Ministral-3-Instruct-8B|Ministral-3-Reasoning-8B)
      echo "2 2 0.88" ;;
    *) echo "No A100 engine settings for $1" >&2; return 2 ;;
  esac
}

run_worker() {
  local ordinal="$1"
  local worker_spec="$2"
  local model="${worker_spec%%|*}"
  local modes_token="${worker_spec#*|}"
  local modes="${modes_token//+/,}"
  local revision
  local request_batch_size max_num_seqs gpu_utilization
  [[ "${model}" != "${worker_spec}" && -n "${model}" && -n "${modes_token}" ]]
  revision="$(PYTHONPATH="${repo}/src" python3 -c \
    'import sys; from realistic_niah_v3_1.spec import MODEL_REVISIONS; print(MODEL_REVISIONS[sys.argv[1]])' \
    "${model}")"
  read -r request_batch_size max_num_seqs gpu_utilization \
    < <(engine_settings_for "${model}")
  env CUDA_VISIBLE_DEVICES="${allocated_gpus[${ordinal}]}" \
    python3 "${work_root}/chtc_v31_a100_run_group.py" \
      --stimuli "${run_root}/dataset/stimuli.jsonl" \
      --run-root "${run_root}" --model "${model}" --revision "${revision}" \
      --prompt-modes "${modes}" --cache-dir "${cache}" --repo-root "${repo}" \
      --max-model-len 32768 --gpu-memory-utilization "${gpu_utilization}" \
      --max-num-seqs "${max_num_seqs}" \
      --request-batch-size "${request_batch_size}" \
      > "${partial_meta}/logs/worker${ordinal}.out" \
      2> "${partial_meta}/logs/worker${ordinal}.err"
}

pids=()
for ordinal in 0 1 2 3; do
  run_worker "${ordinal}" "${worker_specs[${ordinal}]}" &
  pids+=("$!")
done

worker_status=0
for pid in "${pids[@]}"; do
  wait "${pid}" || worker_status=1
done
[[ "${worker_status}" -eq 0 ]] || {
  echo "At least one ${group_id} worker failed" >&2
  exit 1
}

for worker_spec in "${worker_specs[@]}"; do
  model="${worker_spec%%|*}"
  modes_token="${worker_spec#*|}"
  IFS='+' read -r -a modes <<< "${modes_token}"
  for mode in "${modes[@]}"; do
    task_id="${model}__${mode}"
    python3 -c \
      'import json,sys; p=json.load(open(sys.argv[1])); assert p["protocol_version"]=="realistic_niah_v3_1"; assert p["completed_requests"]==p["expected_requests"]==3360; assert p["prompt_payload_storage"]=="sha256_only"' \
      "${run_root}/shards/${task_id}/main/run_manifest.json"
  done
done

printf '{"group_id":"%s","tier":"%s","workers":4,"completed_at_utc":"%s"}\n' \
  "${group_id}" "${tier}" "$(date -u +%Y-%m-%dT%H:%M:%SZ)" \
  > "${partial_meta}/completed.json"
