#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 1 ]]; then
  echo "Usage: $0 RUN_ROOT" >&2
  exit 2
fi

run_root="$(readlink -f "$1")"
repo="${REALISTIC_NIAH_REPO_ROOT:-/lambda/nfs/Twist-CoT-Count-Multi-Model-v2/code/Realistic_CoT_NiaH_Count}"
python_bin="${REALISTIC_NIAH_PYTHON:-/home/ubuntu/venvs/realistic-niah-vllm/bin/python}"
audit="${run_root}/orchestration/stimuli_audit.json"
session_prefix="rniah-v2-formal-gpu"

case "${run_root}" in
  /lambda/nfs/Twist-CoT-Count-Multi-Model-v2/runs/realistic_niah_v2/eight_models_formal_*)
    ;;
  *)
    echo "Refusing unexpected run root: ${run_root}" >&2
    exit 2
    ;;
esac

test -x "${python_bin}"
test -s "${run_root}/dataset/stimuli.jsonl"
test -s "${audit}"
test -z "$(git -C "${repo}" status --short)"
"${python_bin}" -c \
  'import json,sys; assert json.load(open(sys.argv[1],encoding="utf-8"))["passed"]' \
  "${audit}"
gpu_count="$(nvidia-smi --query-gpu=index --format=csv,noheader | wc -l)"
if [[ "${gpu_count}" -ne 8 ]]; then
  echo "Expected exactly 8 visible GPUs, found ${gpu_count}" >&2
  exit 2
fi
if tmux list-sessions -F '#{session_name}' 2>/dev/null \
  | grep -q "^${session_prefix}"; then
  echo "Formal tmux workers already exist" >&2
  exit 2
fi

mkdir -p "${run_root}/orchestration"
(
  cd "${repo}"
  PYTHONPATH=src "${python_bin}" \
    scripts/build_realistic_niah_formal_shards.py \
    --json-output "${run_root}/orchestration/formal_shards.json" \
    --tsv-output "${run_root}/orchestration/formal_shards.tsv" \
    > "${run_root}/orchestration/formal_shards.build.log"
)

for gpu_id in $(seq 0 7); do
  session="${session_prefix}${gpu_id}"
  command="cd $(printf '%q' "${repo}") && exec bash scripts/run_realistic_niah_formal_v2_worker.sh $(printf '%q' "${run_root}") ${gpu_id}"
  tmux new-session -d -s "${session}" "${command}"
done

tmux list-sessions -F '#{session_name}' \
  | grep "^${session_prefix}" \
  | sort
