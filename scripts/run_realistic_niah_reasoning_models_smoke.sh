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
  */runs/realistic_niah_v2/reasoning_models_smoke_*)
    ;;
  *)
    echo "SMOKE_ROOT must end in reasoning_models_smoke_*" >&2
    exit 2
    ;;
esac

test -x "${python_bin}"
test -d "${cache}"
test -z "$(git -C "${repo}" status --short)"
export PATH="$(dirname "${python_bin}"):${PATH}"
"${python_bin}" -c \
  'from importlib.metadata import version; from packaging.version import Version; assert version("transformers") == "5.5.3"; assert version("vllm") == "0.25.1"; assert Version(version("mistral-common")) >= Version("1.8.6")'
if ! nvidia-smi --query-gpu=index --format=csv,noheader \
  | tr -d ' ' | grep -qx '0'; then
  echo "Smoke test requires visible GPU 0" >&2
  exit 2
fi

cd "${repo}"
PYTHONPATH=src "${python_bin}" \
  scripts/prepare_realistic_niah_reasoning_models_extension.py \
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

run_checkpoint() {
  local model="$1"
  local revision="$2"
  local modes="$3"
  local request_batch_size="$4"
  local max_num_seqs="$5"
  CUDA_VISIBLE_DEVICES=0 PYTHONPATH=src TOKENIZERS_PARALLELISM=false \
    "${python_bin}" scripts/run_realistic_niah.py \
    "${common_args[@]}" \
    --output-dir "${smoke_root}/models/${model}/smoke" \
    --model "${model}" \
    --revision "${revision}" \
    --prompt-modes "${modes}" \
    --max-num-seqs "${max_num_seqs}" \
    --request-batch-size "${request_batch_size}"
}

run_checkpoint \
  Nemotron-Nano-v2-9B \
  6533e8de2c68e4536bf7c411d7a3ce5734111476 \
  direct,enumeration_index,enumeration_bullet,native_thinking \
  4 4
run_checkpoint \
  Nemotron-3-Nano-4B \
  dfaf35de3e30f1867dd8dbc38a7fc9fb52d3914f \
  direct,enumeration_index,enumeration_bullet,native_thinking \
  6 6
run_checkpoint \
  Granite-3.3-Instruct-8B \
  51dd4bc2ade4059a6bd87649d68aa11e4fb2529b \
  direct,enumeration_index,enumeration_bullet,native_thinking \
  8 8
run_checkpoint \
  Cogito-v1-Preview-8B \
  64c42369b3f322fbffb277bfff146551dd2823cc \
  direct,enumeration_index,enumeration_bullet,native_thinking \
  8 8
run_checkpoint \
  Ministral-3-Instruct-8B \
  5b26027e7b19eeb4b7352e1fed3926375dd2cb4d \
  direct,enumeration_index,enumeration_bullet \
  8 8
run_checkpoint \
  Ministral-3-Reasoning-8B \
  81eaece1948f3875421d9a45bc55487d10e2d894 \
  native_thinking \
  4 4

PYTHONPATH=src "${python_bin}" \
  scripts/audit_realistic_niah_reasoning_models_smoke.py \
  --smoke-root "${smoke_root}" \
  --output "${smoke_root}/orchestration/smoke_audit.json"
