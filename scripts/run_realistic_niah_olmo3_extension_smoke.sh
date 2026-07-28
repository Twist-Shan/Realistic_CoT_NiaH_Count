#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 2 ]]; then
  echo "Usage: $0 SOURCE_FORMAL_RUN_ROOT SMOKE_ROOT" >&2
  exit 2
fi

source_formal_root="$(readlink -f "$1")"
smoke_root="$(readlink -m "$2")"
repo="${REALISTIC_NIAH_REPO_ROOT:-/lambda/nfs/Twist-CoT-Count-Multi-Model-v2/code/Realistic_CoT_NiaH_Count}"
cache="${REALISTIC_NIAH_HF_CACHE:-/lambda/nfs/Twist-CoT-Count-Multi-Model-v2/hf-cache}"
python_bin="${REALISTIC_NIAH_PYTHON:-/home/ubuntu/venvs/realistic-niah-vllm/bin/python}"

case "${smoke_root}" in
  */runs/realistic_niah_v2/olmo3_7b_smoke_*)
    ;;
  *)
    echo "SMOKE_ROOT must end in realistic_niah_v2/olmo3_7b_smoke_*" >&2
    exit 2
    ;;
esac

test -x "${python_bin}"
test -d "${cache}"
test -z "$(git -C "${repo}" status --short)"
"${python_bin}" -c \
  'from importlib.metadata import version; import vllm; assert version("transformers") == "5.5.3"; assert version("vllm") == "0.25.1"'
if ! nvidia-smi --query-gpu=index --format=csv,noheader \
  | tr -d ' ' | grep -qx '0'; then
  echo "Smoke test requires visible GPU 0" >&2
  exit 2
fi

cd "${repo}"
PYTHONPATH=src "${python_bin}" \
  scripts/prepare_realistic_niah_olmo3_extension.py \
  --source-formal-run-root "${source_formal_root}" \
  --run-root "${smoke_root}" \
  --repo-root "${repo}"

common_args=(
  --stimuli "${smoke_root}/dataset/stimuli.jsonl"
  --passage-lengths 2000,20000
  --needle-counts 6,30
  --seeds 1234
  --query-layout cue_before_query_after
  --cache-dir "${cache}"
  --repo-root "${repo}"
  --tensor-parallel-size 1
  --max-model-len 32768
  --gpu-memory-utilization 0.90
  --require-clean-git
)

CUDA_VISIBLE_DEVICES=0 PYTHONPATH=src TOKENIZERS_PARALLELISM=false \
  "${python_bin}" scripts/run_realistic_niah.py \
  "${common_args[@]}" \
  --output-dir "${smoke_root}/models/Olmo3-7B-Instruct/smoke" \
  --model Olmo3-7B-Instruct \
  --revision 6e5971d9eba42665f5bd5a0fcf047f299ce1dccc \
  --prompt-modes direct,enumeration_index,enumeration_bullet \
  --max-num-seqs 8 \
  --request-batch-size 8

CUDA_VISIBLE_DEVICES=0 PYTHONPATH=src TOKENIZERS_PARALLELISM=false \
  "${python_bin}" scripts/run_realistic_niah.py \
  "${common_args[@]}" \
  --output-dir "${smoke_root}/models/Olmo3-7B-Think/smoke" \
  --model Olmo3-7B-Think \
  --revision d97e442d7cc678210054dbcc9b440894d62c89a4 \
  --prompt-modes native_thinking \
  --max-num-seqs 4 \
  --request-batch-size 6

PYTHONPATH=src "${python_bin}" \
  scripts/audit_realistic_niah_olmo3_smoke.py \
  --smoke-root "${smoke_root}" \
  --output "${smoke_root}/orchestration/smoke_audit.json"
