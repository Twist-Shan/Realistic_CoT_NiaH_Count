#!/usr/bin/env bash
set -euo pipefail

VENV_DIR="${REALISTIC_NIAH_VENV:-/home/ubuntu/venvs/realistic-niah-vllm}"
PERSIST_ROOT="${REALISTIC_NIAH_PERSIST_ROOT:-/lambda/nfs/Twist-CoT-Count-Multi-Model}"
SITE_PACKAGES="${VENV_DIR}/lib/python3.10/site-packages"
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"

if [[ ! -x "${VENV_DIR}/bin/python" ]]; then
  echo "Python environment not found: ${VENV_DIR}" >&2
  exit 1
fi

# vLLM 0.25.1 ships an auxiliary CUDA 13 extension alongside its CUDA 12.8
# PyTorch wheel. Lambda's driver supports both, but the auxiliary runtime
# directory must be visible to the dynamic loader.
export LD_LIBRARY_PATH="${SITE_PACKAGES}/nvidia/cu13/lib:${SITE_PACKAGES}/nvidia/cuda_runtime/lib:${LD_LIBRARY_PATH:-}"
export PATH="${VENV_DIR}/bin:${PATH}"
export HF_HOME="${PERSIST_ROOT}/hf-cache"
export HF_TOKEN_PATH="${HF_TOKEN_PATH:-${HOME}/.cache/huggingface/token}"
export PIP_CACHE_DIR="${PERSIST_ROOT}/pip-cache"
export TOKENIZERS_PARALLELISM=false
export PYTHONPATH="${REPO_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"

exec "${VENV_DIR}/bin/python" "$@"
