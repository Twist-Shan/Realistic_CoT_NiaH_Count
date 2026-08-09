#!/usr/bin/env bash
set -Eeuo pipefail

CAMPAIGN_ROOT=/lambda/nfs/CoT-Non-thinking-v4/runs/run_20260808_v4_4_first_span_locator_topk_k1_2_4_8_16_32
RESULT_ROOT=${1:-"$CAMPAIGN_ROOT/m10_full"}
MODE=${2:-full}
SOURCE_DATASET="$CAMPAIGN_ROOT/full/dataset/first_span_confirmation"
REPO_ROOT=/lambda/nfs/CoT-Non-thinking-v4/repo
PY=/lambda/nfs/CoT-Non-thinking-v4/venv/bin/python
HF_CACHE=/lambda/nfs/CoT-Non-thinking-v4/hf-cache
BASE_CONFIG="$CAMPAIGN_ROOT/inputs/experiment_first_span.json"
SELECTION="$CAMPAIGN_ROOT/inputs/selection_first_span_absolute_mass.json"
CAUSAL_CONFIG="$REPO_ROOT/configs/realistic_niah_v4_causal_v2.json"
RUNNER="$CAMPAIGN_ROOT/code/run_realistic_niah_v4_causal_v2_first_span.py"
DATASET_ROOT="$RESULT_ROOT/dataset/first_span_confirmation"
STIMULI="$DATASET_ROOT/stimuli_v4_4_causal_v2.jsonl"
LOG_ROOT="$RESULT_ROOT/logs"
mkdir -p "$DATASET_ROOT" "$LOG_ROOT"
if [[ ! -s "$STIMULI" ]]; then cp -a "$SOURCE_DATASET/." "$DATASET_ROOT/"; fi
cd "$REPO_ROOT"
extra=()
if [[ "$MODE" == smoke ]]; then extra+=(--smoke); fi

for model in Qwen3-8B Gemma4-E4B; do
  ranking="$CAMPAIGN_ROOT/inputs/${model}.first_span_absolute_mass_rankings.json"
  control="$CAMPAIGN_ROOT/inputs/${model}.M10_matched_controls.json"
  log="$LOG_ROOT/${model}_first_span_M10_full.log"
  printf '[%s] START %s M10 full\n' "$(date -u +%FT%TZ)" "$model" | tee -a "$RESULT_ROOT/supervisor.log"
  env HF_HOME="$HF_CACHE" TOKENIZERS_PARALLELISM=false \
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    PYTHONPATH="$CAMPAIGN_ROOT/code:$REPO_ROOT/src" \
    V44_M10_CONTROL_JSON="$control" \
    "$PY" "$RUNNER" --stage ablation --phase confirmation \
      --run-root "$RESULT_ROOT" --stimuli "$STIMULI" --model "$model" \
      --config "$BASE_CONFIG" --causal-config "$CAUSAL_CONFIG" \
      --cache-dir "$HF_CACHE" --device-map auto --repo-root "$REPO_ROOT" \
      --generation-max-new-tokens 16 --head-rankings "$ranking" \
      --selection-json "$SELECTION" "${extra[@]}" >"$log" 2>&1
  printf '[%s] COMPLETE %s M10 full\n' "$(date -u +%FT%TZ)" "$model" | tee -a "$RESULT_ROOT/supervisor.log"
done

touch "$RESULT_ROOT/first_span_M10_${MODE}.complete"
