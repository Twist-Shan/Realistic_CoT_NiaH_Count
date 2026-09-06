#!/usr/bin/env bash
set -euo pipefail

ROOT=${V6_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}
PYTHON=${V6_PYTHON:-$ROOT/.venv/bin/python}
CACHE=${V6_CACHE:-$ROOT/.cache/huggingface}
RUN_BASE=${V6_RUN_BASE:-$ROOT/work/realistic_niah_v6}
REPLACEMENT_POOL=${V6_REPLACEMENT_POOL:-$RUN_BASE/replacement_seed_pool}
ANSWER_TRACE_AMENDMENT_POOL=${V6_ANSWER_TRACE_AMENDMENT_POOL:-}
ANSWER_TRACE_AMENDMENT_POLICY=${V6_ANSWER_TRACE_AMENDMENT_POLICY:-}
ANSWER_TRACE_COHORT_AMENDMENT=${V6_ANSWER_TRACE_COHORT_AMENDMENT:-}
GPU_INDEX=${V6_ANSWER_TRACE_GPU_INDEX:-0}
QUEUE_ROOT=$RUN_BASE/queue_logs
LOG=$QUEUE_ROOT/answer_trace_extension.log
COMPLETE=$QUEUE_ROOT/answer_trace_extension.COMPLETE
REPORT_ROOT=$RUN_BASE/answer_trace_extension_report
mkdir -p "$QUEUE_ROOT" "$REPORT_ROOT"
exec > >(tee -a "$LOG") 2>&1

wait_for_pass() {
  local marker=$1
  local label=$2
  echo "[$(date --iso-8601=seconds)] WAIT $label marker=$marker"
  while [[ ! -s "$marker" ]]; do sleep 30; done
  grep -qx PASS "$marker" || { echo "$label is not PASS" >&2; exit 1; }
  echo "[$(date --iso-8601=seconds)] READY $label"
}

wait_for_pass "$QUEUE_ROOT/Qwen3-8B_confirmation.COMPLETE" \
  "Qwen primary confirmation"
wait_for_pass "$QUEUE_ROOT/Gemma4-E4B_confirmation.COMPLETE" \
  "Gemma primary confirmation"
wait_for_pass "$QUEUE_ROOT/index_item_end_anchor_sensitivity.COMPLETE" \
  "index item-end sensitivity"

common_env=(env V6_ROOT="$ROOT" V6_PYTHON="$PYTHON" V6_CACHE="$CACHE"
  V6_RUN_BASE="$RUN_BASE" V6_REPLACEMENT_POOL="$REPLACEMENT_POOL"
  HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1)

for model in Qwen3-8B Gemma4-E4B; do
  for mode in index bullet; do
    echo "[$(date --iso-8601=seconds)] START $model $mode answer/trace extension"
    cell_env=("${common_env[@]}")
    if [[ "$model" == Gemma4-E4B && "$mode" == bullet && \
          -n "$ANSWER_TRACE_COHORT_AMENDMENT" ]]; then
      test -n "$ANSWER_TRACE_AMENDMENT_POOL"
      test -n "$ANSWER_TRACE_AMENDMENT_POLICY"
      cell_env+=(
        V6_REPLACEMENT_POOL="$ANSWER_TRACE_AMENDMENT_POOL"
        V6_REPLACEMENT_POLICY="$ANSWER_TRACE_AMENDMENT_POLICY"
        V6_ANSWER_TRACE_COHORT_AMENDMENT="$ANSWER_TRACE_COHORT_AMENDMENT"
      )
    fi
    "${cell_env[@]}" bash \
      "$ROOT/scripts/supervise_realistic_niah_v6_answer_trace_extension.sh" \
      "$mode" "$model" "$GPU_INDEX"
    echo "[$(date --iso-8601=seconds)] PASS $model $mode answer/trace extension"
  done
done

cd "$ROOT"
"$PYTHON" scripts/build_realistic_niah_v6_answer_trace_extension_report.py \
  --run-root "$RUN_BASE" --output-root "$REPORT_ROOT"
grep -qx PASS "$REPORT_ROOT/report.COMPLETE"
printf 'PASS\n' >"$COMPLETE"
echo "[$(date --iso-8601=seconds)] COMPLETE V6 answer/trace extension"
