#!/usr/bin/env bash
set -Eeuo pipefail

if [[ $# -ne 5 ]]; then
  echo "Usage: $0 RUN_ROOT VENV_PYTHON HF_CACHE QWEN_RANKINGS GEMMA_RANKINGS" >&2
  exit 2
fi

RUN_ROOT=$1
VENV_PYTHON=$2
HF_CACHE=$3
QWEN_RANKINGS_SOURCE=$4
GEMMA_RANKINGS_SOURCE=$5
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd -- "$SCRIPT_DIR/.." && pwd)
BASE_CONFIG="$REPO_ROOT/configs/realistic_niah_v4_4_full_span_topk_confirmation.json"
CAUSAL_CONFIG="$REPO_ROOT/configs/realistic_niah_v4_causal_v2.json"
SELECTION="$REPO_ROOT/configs/realistic_niah_v4_4_full_span_topk_confirmation_selection.json"
DATASET_ROOT="$RUN_ROOT/dataset/full_span_topk_confirmation"
STIMULI="$DATASET_ROOT/stimuli_v4_4_causal_v2.jsonl"
INPUT_ROOT="$RUN_ROOT/inputs"
LOG_ROOT="$RUN_ROOT/logs"
STATUS="$RUN_ROOT/full_span_topk.status"
COMPLETE="$RUN_ROOT/full_span_topk.complete"
SUPERVISOR_LOG="$LOG_ROOT/full_span_topk_supervisor.log"

cd "$REPO_ROOT"
for path in "$VENV_PYTHON" "$BASE_CONFIG" "$CAUSAL_CONFIG" "$SELECTION" \
  "$QWEN_RANKINGS_SOURCE" "$GEMMA_RANKINGS_SOURCE"; do
  [[ -e "$path" ]] || { echo "Required path is missing: $path" >&2; exit 2; }
done
mkdir -p "$DATASET_ROOT" "$INPUT_ROOT" "$LOG_ROOT"
touch "$SUPERVISOR_LOG"

QWEN_RANKINGS="$INPUT_ROOT/Qwen3-8B.full_span_head_rankings.json"
GEMMA_RANKINGS="$INPUT_ROOT/Gemma4-E4B.full_span_head_rankings.json"
cp -- "$QWEN_RANKINGS_SOURCE" "$QWEN_RANKINGS"
cp -- "$GEMMA_RANKINGS_SOURCE" "$GEMMA_RANKINGS"
chmod 0444 "$QWEN_RANKINGS" "$GEMMA_RANKINGS"
printf '%s  %s\n' \
  '7caa1362c675ea47818cf0420d78c66963a2cca3aca17639e16327fb14434492' \
  "$QWEN_RANKINGS" \
  '52646c9bec4d90336f469c46aecd8ffbaf54008648f7aef45f56215984744de7' \
  "$GEMMA_RANKINGS" | sha256sum -c -

timestamp() { date -u +'%Y-%m-%dT%H:%M:%SZ'; }
write_status() {
  local state=$1 model=$2 stage=$3 log_path=$4 temporary="$STATUS.tmp.$$"
  printf '{"state":"%s","model":"%s","stage":"%s","updated_utc":"%s","log":"%s"}\n' \
    "$state" "$model" "$stage" "$(timestamp)" "$log_path" > "$temporary"
  mv -f "$temporary" "$STATUS"
}
CURRENT_MODEL=campaign
CURRENT_STAGE=initializing
CURRENT_LOG="$SUPERVISOR_LOG"
on_error() {
  local rc=$?
  write_status "FAILED:$rc" "$CURRENT_MODEL" "$CURRENT_STAGE" "$CURRENT_LOG"
  printf '[%s] FAILED rc=%s model=%s stage=%s\n' \
    "$(timestamp)" "$rc" "$CURRENT_MODEL" "$CURRENT_STAGE" | tee -a "$SUPERVISOR_LOG" >&2
  exit "$rc"
}
trap on_error ERR

run_python() {
  env HF_HOME="$HF_CACHE" \
    HF_TOKEN_PATH="${HF_TOKEN_PATH:-${HOME}/.cache/huggingface/token}" \
    TOKENIZERS_PARALLELISM=false \
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    PYTHONPATH="$REPO_ROOT/src" \
    "$VENV_PYTHON" "$@"
}
run_logged() {
  local model=$1 stage=$2 log_path=$3
  shift 3
  CURRENT_MODEL=$model; CURRENT_STAGE=$stage; CURRENT_LOG=$log_path
  write_status RUNNING "$model" "$stage" "$log_path"
  printf '[%s] START model=%s stage=%s\n' "$(timestamp)" "$model" "$stage" \
    | tee -a "$SUPERVISOR_LOG" "$log_path"
  run_python "$@" >> "$log_path" 2>&1
  write_status COMPLETE "$model" "$stage" "$log_path"
  printf '[%s] COMPLETE model=%s stage=%s\n' "$(timestamp)" "$model" "$stage" \
    | tee -a "$SUPERVISOR_LOG" "$log_path"
}

if [[ ! -s "$STIMULI" ]]; then
  run_logged campaign freeze "$LOG_ROOT/freeze.log" \
    "$REPO_ROOT/scripts/freeze_realistic_niah_v4_causal_v2.py" \
      --output-dir "$DATASET_ROOT" --config "$BASE_CONFIG" --cache-dir "$HF_CACHE"
else
  (cd "$DATASET_ROOT" && sha256sum -c SHA256SUMS_v4_4_causal_v2) \
    >> "$LOG_ROOT/freeze.log" 2>&1
fi

MODELS=(Qwen3-8B Gemma4-E4B)
RANKINGS=("$QWEN_RANKINGS" "$GEMMA_RANKINGS")
for index in 0 1; do
  model=${MODELS[$index]}
  ranking=${RANKINGS[$index]}
  run_logged "$model" ablation-full-span-topk \
    "$LOG_ROOT/${model}_full_span_topk.log" \
    "$REPO_ROOT/scripts/run_realistic_niah_v4_causal_v2.py" \
      --stage ablation --phase confirmation --run-root "$RUN_ROOT" \
      --stimuli "$STIMULI" --model "$model" --config "$BASE_CONFIG" \
      --causal-config "$CAUSAL_CONFIG" --cache-dir "$HF_CACHE" \
      --device-map auto --repo-root "$REPO_ROOT" \
      --generation-max-new-tokens 16 --head-rankings "$ranking" \
      --selection-json "$SELECTION"
done

run_logged campaign audit "$LOG_ROOT/audit.log" \
  "$REPO_ROOT/scripts/audit_realistic_niah_v4_4_ablation_seed_extrapolation.py" \
    --run-root "$RUN_ROOT" --repo-root "$REPO_ROOT" --stimuli "$STIMULI" \
    --selection-json "$SELECTION" --qwen-rankings "$QWEN_RANKINGS" \
    --gemma-rankings "$GEMMA_RANKINGS"

temporary="$COMPLETE.tmp.$$"
printf '{"status":"complete","completed_utc":"%s","audit":"%s"}\n' \
  "$(timestamp)" \
  "$RUN_ROOT/audit/ablation_seed_extrapolation/ablation_seed_extrapolation_audit.json" \
  > "$temporary"
mv -f "$temporary" "$COMPLETE"
write_status COMPLETE campaign all "$SUPERVISOR_LOG"
printf '[%s] CAMPAIGN COMPLETE\n' "$(timestamp)" | tee -a "$SUPERVISOR_LOG"
