#!/usr/bin/env bash
set -Eeuo pipefail

usage() {
  cat <<'EOF'
Usage:
  launch_realistic_niah_v4_4_correct_interventions.sh \
    RUN_ROOT VENV_PYTHON HF_CACHE QWEN_SOURCE_RUN GEMMA_SOURCE_RUN \
    ABLATION_CONFIRMATION_SOURCE_RUN

Runs the independent V4.4 extension sequentially on one GPU. Fresh seeds are
scanned in ascending order using only clean baseline correctness. The runner
then patches only the minimum additional clean-correct receiver/donor pairs
needed to reach five seed clusters per model/k/direction. For ablation it runs
top-n=1..32 once on every count-1..5 example in a shared fresh-seed prefix and
derives the clean-correct population as an exact subset of those rows. It
reports candidates but does not freeze or confirm any top-n.
EOF
}

if [[ $# -ne 6 ]]; then
  usage >&2
  exit 2
fi

RUN_ROOT=$1
VENV_PYTHON=$2
HF_CACHE=$3
QWEN_SOURCE_RUN=$4
GEMMA_SOURCE_RUN=$5
ABLATION_SOURCE_RUN=$6
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd -- "$SCRIPT_DIR/.." && pwd)
BASE_CONFIG="$REPO_ROOT/configs/realistic_niah_v4.json"
CAUSAL_CONFIG="$REPO_ROOT/configs/realistic_niah_v4_causal_v2.json"
DEFINITION="$REPO_ROOT/configs/realistic_niah_v4_4_correct_interventions.json"
DATASET_ROOT="$RUN_ROOT/dataset/correct_interventions"
STIMULI="$DATASET_ROOT/stimuli_v4_4_causal_v2.jsonl"
LOG_ROOT="$RUN_ROOT/logs/correct_interventions"
STATUS="$RUN_ROOT/correct_interventions.status"
COMPLETE="$RUN_ROOT/correct_interventions.complete"

for path in "$VENV_PYTHON" "$BASE_CONFIG" "$CAUSAL_CONFIG" "$DEFINITION" \
  "$QWEN_SOURCE_RUN" "$GEMMA_SOURCE_RUN" "$ABLATION_SOURCE_RUN"; do
  if [[ ! -e "$path" ]]; then
    echo "Required path is missing: $path" >&2
    exit 2
  fi
done
if [[ -n "$(git -C "$REPO_ROOT" status --porcelain)" ]]; then
  echo "Formal correct-intervention run requires a clean repository" >&2
  git -C "$REPO_ROOT" status --short >&2
  exit 2
fi
mkdir -p "$DATASET_ROOT" "$LOG_ROOT"

timestamp() { date -u +'%Y-%m-%dT%H:%M:%SZ'; }

write_status() {
  local state=$1 model=$2 stage=$3 log_path=$4
  printf '{"state":"%s","model":"%s","stage":"%s","updated_utc":"%s","log":"%s"}\n' \
    "$state" "$model" "$stage" "$(timestamp)" "$log_path" > "$STATUS"
}

CURRENT_MODEL=campaign
CURRENT_STAGE=initializing
CURRENT_LOG="$LOG_ROOT/supervisor.log"
on_error() {
  local rc=$?
  write_status "FAILED:$rc" "$CURRENT_MODEL" "$CURRENT_STAGE" "$CURRENT_LOG"
  printf '[%s] FAILED rc=%s model=%s stage=%s\n' \
    "$(timestamp)" "$rc" "$CURRENT_MODEL" "$CURRENT_STAGE" \
    | tee -a "$LOG_ROOT/supervisor.log" >&2
  exit "$rc"
}
trap on_error ERR

run_python() {
  env \
    HF_HOME="$HF_CACHE" \
    HF_TOKEN_PATH="${HF_TOKEN_PATH:-${HOME}/.cache/huggingface/token}" \
    TOKENIZERS_PARALLELISM=false \
    PYTHONPATH="$REPO_ROOT/src" \
    "$VENV_PYTHON" "$@"
}

run_logged() {
  local model=$1 stage=$2 log_path=$3
  shift 3
  CURRENT_MODEL=$model
  CURRENT_STAGE=$stage
  CURRENT_LOG=$log_path
  write_status RUNNING "$model" "$stage" "$log_path"
  printf '[%s] START model=%s stage=%s\n' "$(timestamp)" "$model" "$stage" \
    | tee -a "$LOG_ROOT/supervisor.log" "$log_path"
  run_python "$@" >> "$log_path" 2>&1
  write_status COMPLETE "$model" "$stage" "$log_path"
  printf '[%s] COMPLETE model=%s stage=%s\n' "$(timestamp)" "$model" "$stage" \
    | tee -a "$LOG_ROOT/supervisor.log" "$log_path"
}

latest_complete_root() {
  local source_root=$1 model=$2 family=$3 phase=$4
  local parent="$source_root/$model/numeric/causal_v2/$family"
  find "$parent" -maxdepth 2 -type f -path "*/${phase}_*/complete.json" \
    -printf '%T@ %h\n' | sort -n | tail -1 | cut -d' ' -f2-
}

cd "$REPO_ROOT"
if [[ ! -s "$STIMULI" ]]; then
  run_logged campaign freeze "$LOG_ROOT/freeze.log" \
    "$REPO_ROOT/scripts/freeze_realistic_niah_v4_4_correct_interventions.py" \
      --output-dir "$DATASET_ROOT" \
      --base-config "$BASE_CONFIG" \
      --definition "$DEFINITION" \
      --cache-dir "$HF_CACHE"
else
  CURRENT_MODEL=campaign
  CURRENT_STAGE=verify-frozen-stimuli
  CURRENT_LOG="$LOG_ROOT/freeze.log"
  (cd "$DATASET_ROOT" && sha256sum -c SHA256SUMS_v4_4_causal_v2) \
    >> "$CURRENT_LOG" 2>&1
fi

MODELS=(Qwen3-8B Gemma4-E4B)
SOURCES=("$QWEN_SOURCE_RUN" "$GEMMA_SOURCE_RUN")
for index in 0 1; do
  model=${MODELS[$index]}
  source=${SOURCES[$index]}
  prompt_screen=$(latest_complete_root "$source" "$model" prompt_patching screen)
  prompt_confirmation=$(
    latest_complete_root "$source" "$model" prompt_patching confirmation
  )
  answer_screen=$(latest_complete_root "$source" "$model" answer_patching screen)
  answer_confirmation=$(
    latest_complete_root "$source" "$model" answer_patching confirmation
  )
  ranking_root=$(latest_complete_root "$source" "$model" head_rankings discovery)
  ablation_confirmation=$(
    latest_complete_root \
      "$ABLATION_SOURCE_RUN" "$model" answer_query_head_ablation confirmation
  )
  ablation_discovery=$(
    latest_complete_root \
      "$source" "$model" answer_query_head_ablation screen
  )
  prompt_selection="$prompt_screen/selection/prompt_patching_selection.json"
  answer_selection="$answer_screen/selection/answer_patching_selection.json"
  prompt_detail="$prompt_confirmation/detail.csv.gz"
  answer_detail="$answer_confirmation/detail.csv.gz"
  ablation_detail="$ablation_confirmation/detail.csv.gz"
  ablation_discovery_detail="$ablation_discovery/detail.csv.gz"
  rankings="$ranking_root/head_phenotype_rankings.json"
  for path in "$prompt_selection" "$answer_selection" "$prompt_detail" \
    "$answer_detail" "$ablation_discovery_detail" "$ablation_detail" \
    "$rankings"; do
    if [[ ! -s "$path" ]]; then
      echo "Resolved source artifact is missing or empty: $path" >&2
      exit 2
    fi
  done
  run_logged "$model" correct-interventions "$LOG_ROOT/${model}.log" \
    "$REPO_ROOT/scripts/run_realistic_niah_v4_4_correct_interventions.py" \
      --run-root "$RUN_ROOT" \
      --stimuli "$STIMULI" \
      --model "$model" \
      --base-config "$BASE_CONFIG" \
      --causal-config "$CAUSAL_CONFIG" \
      --definition "$DEFINITION" \
      --prompt-selection "$prompt_selection" \
      --answer-selection "$answer_selection" \
      --prompt-confirmation-detail "$prompt_detail" \
      --answer-confirmation-detail "$answer_detail" \
      --ablation-discovery-detail "$ablation_discovery_detail" \
      --ablation-confirmation-detail "$ablation_detail" \
      --head-rankings "$rankings" \
      --cache-dir "$HF_CACHE" \
      --device-map auto \
      --repo-root "$REPO_ROOT" \
      --generation-max-new-tokens 16
done

run_logged campaign audit "$LOG_ROOT/audit.log" \
  "$REPO_ROOT/scripts/audit_realistic_niah_v4_4_correct_interventions.py" \
    --run-root "$RUN_ROOT" \
    --definition "$DEFINITION"

printf '{"status":"complete","completed_utc":"%s","audit":"%s"}\n' \
  "$(timestamp)" "$RUN_ROOT/audit/correct_interventions/audit.json" > "$COMPLETE"
write_status COMPLETE campaign all "$LOG_ROOT/supervisor.log"
printf '[%s] CAMPAIGN COMPLETE\n' "$(timestamp)" | tee -a "$LOG_ROOT/supervisor.log"
