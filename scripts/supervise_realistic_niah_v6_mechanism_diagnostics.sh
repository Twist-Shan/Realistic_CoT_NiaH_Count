#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 2 ]]; then
  echo "usage: $0 <Qwen3-8B|Gemma4-E4B> <gpu-index>" >&2
  exit 2
fi
MODEL=$1
GPU_INDEX=$2
case "$MODEL" in
  Qwen3-8B) SOURCE_LAYER=19 ;;
  Gemma4-E4B) SOURCE_LAYER=16 ;;
  *) echo "unsupported model: $MODEL" >&2; exit 2 ;;
esac
if [[ ! "$GPU_INDEX" =~ ^[0-9]+$ ]]; then
  echo "gpu-index must be non-negative" >&2
  exit 2
fi

ROOT=${V6_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}
PYTHON=${V6_PYTHON:-$ROOT/.venv/bin/python}
CACHE=${V6_CACHE:-$ROOT/.cache/huggingface}
RUN_BASE=${V6_RUN_BASE:-$ROOT/work/realistic_niah_v6}
PROTOCOL=$ROOT/configs/realistic_niah_v6_mechanism_diagnostic_extension_v1.json
GLOBAL_ROOT=$RUN_BASE/mechanism_diagnostic_extension
LOG_ROOT=$GLOBAL_ROOT/logs/$MODEL
LOCK_ROOT=$GLOBAL_ROOT/locks
mkdir -p "$LOG_ROOT" "$LOCK_ROOT"
cd "$ROOT"

for path in "$PYTHON" "$PROTOCOL"; do
  [[ -s "$path" ]] || { echo "missing mechanism diagnostic input: $path" >&2; exit 4; }
done

exec 9>"$LOCK_ROOT/$MODEL.lock"
if ! flock -n 9; then
  echo "another $MODEL mechanism diagnostic supervisor owns the lock" >&2
  exit 75
fi

run_logged() {
  local name=$1
  shift
  {
    echo "[$(date --iso-8601=seconds)] START $name"
    printf 'COMMAND env CUDA_VISIBLE_DEVICES=%q' "$GPU_INDEX"
    printf ' %q' "$@"
    printf '\n'
    env CUDA_VISIBLE_DEVICES="$GPU_INDEX" HF_HUB_OFFLINE=1 \
      TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false "$@"
    echo "[$(date --iso-8601=seconds)] PASS $name"
  } 2>&1 | tee "$LOG_ROOT/$name.log"
}

prompt_mode() {
  case "$1" in
    index) printf 'enumeration_index\n' ;;
    bullet) printf 'enumeration_bullet\n' ;;
    *) return 2 ;;
  esac
}

phase_role() {
  case "$1" in
    discovery) printf 'development\n' ;;
    confirmation) printf 'confirmation\n' ;;
    *) return 2 ;;
  esac
}

phase_count() {
  case "$1" in
    discovery) printf '20\n' ;;
    confirmation) printf '10\n' ;;
    *) return 2 ;;
  esac
}

manifest_has_shards() {
  local manifest=$1
  local expected=$2
  [[ -s "$manifest" ]] || return 1
  "$PYTHON" -c \
    'import json,sys; x=json.load(open(sys.argv[1])); raise SystemExit(0 if int(x.get("completed_shards",-1))==int(sys.argv[2]) else 1)' \
    "$manifest" "$expected"
}

run_targeted_likelihood() {
  local mode=$1
  local phase=$2
  local prompt model_root config generations discovery_registry registry freeze
  local selection selected_k plan output expected
  prompt=$(prompt_mode "$mode")
  model_root=$RUN_BASE/$prompt/$MODEL
  config=$ROOT/configs/realistic_niah_v6_${prompt}.json
  generations=$model_root/generation/generations.jsonl
  discovery_registry=$model_root/replacement/discovery/selected_cells.jsonl
  registry=$model_root/replacement/$phase/selected_cells.jsonl
  freeze=$model_root/freeze/confirmation_freeze.json
  selection=$model_root/causal/targeted_retrieval/discovery_formal/analysis/selection.json
  expected=$(phase_count "$phase")
  for path in "$config" "$generations" "$discovery_registry" "$registry" "$selection"; do
    [[ -s "$path" ]] || { echo "missing targeted likelihood input: $path" >&2; exit 4; }
  done
  selected_k=$($PYTHON -c \
    'import json,sys; print(int(json.load(open(sys.argv[1]))["selected_k"]))' \
    "$selection")
  plan=$model_root/causal/targeted_retrieval/discovery_formal/plans/k$selected_k/retrieval_anchor_bank_plan.csv
  [[ -s "$plan" ]] || { echo "missing frozen K plan: $plan" >&2; exit 4; }
  output=$model_root/causal/mechanism_diagnostic_extension/targeted_city_likelihood/$phase
  mkdir -p "$output"
  if [[ -s "$output/analysis/claim_gates.json" ]]; then
    echo "[$(date --iso-8601=seconds)] REUSE ${mode}_${phase}_targeted_city_likelihood"
    return
  fi
  local scope_args=()
  if [[ "$mode" == bullet ]]; then
    # The legacy V5 continuation registry labels the V6 Bullet P0 item-end
    # anchor as secondary.  The frozen V6 targeted-retrieval runners already
    # opt into that exact registered scope; carry the same vocabulary adapter
    # into this continuous readout without broadening the V6 panel.
    scope_args+=(--include-secondary)
  fi
  local wrapper=(
    "$PYTHON" scripts/run_realistic_niah_v6_causal.py
    --v6-config "$config" --model-label "$MODEL" --phase "$phase"
    --cohort-registry "$registry"
  )
  if [[ "$phase" == confirmation ]]; then
    [[ -s "$freeze" ]] || { echo "missing confirmation freeze: $freeze" >&2; exit 4; }
    wrapper+=(--confirmation-freeze "$freeze")
    wrapper+=(--causal-membership-registry "$discovery_registry")
  fi
  wrapper+=(-- causal-heads)
  if manifest_has_shards "$output/trials/manifest.json" "$((expected * 5))"; then
    echo "[$(date --iso-8601=seconds)] REUSE ${mode}_${phase}_targeted_city_likelihood trials"
  else
    run_logged "${mode}_${phase}_targeted_city_likelihood" \
      "${wrapper[@]}" \
      --model "$MODEL" --cache-dir "$CACHE" --device-map auto \
      --torch-dtype bfloat16 --attention-backend sdpa \
      --generations "$generations" --plan "$plan" \
      --output "$output/trials" --limit "$expected" "${scope_args[@]}"
  fi
  run_logged "${mode}_${phase}_targeted_city_likelihood_analysis" \
    "$PYTHON" scripts/analyze_realistic_niah_v6_targeted_city_likelihood.py \
    --trials "$output/trials" --phase "$phase" --expected-seeds "$expected" \
    --output "$output/analysis"
}

run_local_terminal() {
  local phase=$1
  local prompt=enumeration_bullet
  local model_root=$RUN_BASE/$prompt/$MODEL
  local config=$ROOT/configs/realistic_niah_v6_${prompt}.json
  local mechanism=$model_root/freeze/mechanism_frozen_confirmation.json
  local generations=$model_root/generation/generations.jsonl
  local registry=$model_root/replacement/$phase/selected_cells.jsonl
  local freeze=$model_root/freeze/confirmation_freeze.json
  local panel=$model_root/causal/targeted_retrieval/${phase}_formal/final_transition_panel/mode_panel.jsonl
  local role expected output
  role=$(phase_role "$phase")
  expected=$(phase_count "$phase")
  output=$model_root/causal/mechanism_diagnostic_extension/local_terminal_bridge/$phase
  for path in "$config" "$mechanism" "$generations" "$registry" "$panel"; do
    [[ -s "$path" ]] || { echo "missing local terminal input: $path" >&2; exit 4; }
  done
  local wrapper=(
    "$PYTHON" scripts/run_realistic_niah_v6_kernel.py
    --target local-terminal-token-state-bridge
    --v6-config "$config" --phase "$phase" --cohort-registry "$registry"
  )
  if [[ "$phase" == confirmation ]]; then
    [[ -s "$freeze" ]] || { echo "missing confirmation freeze: $freeze" >&2; exit 4; }
    wrapper+=(--confirmation-freeze "$freeze")
  fi
  wrapper+=(--)
  if [[ -s "$output/analysis/claim_gates.json" ]]; then
    echo "[$(date --iso-8601=seconds)] REUSE bullet_${phase}_local_terminal_bridge"
    return
  fi
  if manifest_has_shards "$output/trials/manifest.json" "$expected"; then
    echo "[$(date --iso-8601=seconds)] REUSE bullet_${phase}_local_terminal_bridge trials"
  else
    run_logged "bullet_${phase}_local_terminal_bridge" \
      "${wrapper[@]}" \
      --mechanism-config "$mechanism" --model "$MODEL" --cache-dir "$CACHE" \
      --device-map auto --torch-dtype bfloat16 --attention-backend sdpa \
      --generations "$generations" --seed-role "$role" \
      --anchor-registry "$panel" --layer "$SOURCE_LAYER" \
      --max-new-tokens 32 --resume --output "$output/trials"
  fi
  run_logged "bullet_${phase}_local_terminal_bridge_analysis" \
    "$PYTHON" scripts/analyze_realistic_niah_v5_local_terminal_token_state_bridge.py \
    --input "$output/trials" --phase "$phase" --output "$output/analysis"
}

run_decode_aligned_carrier() {
  local phase=$1
  [[ "$MODEL" == Gemma4-E4B ]] || return 0
  local prompt=enumeration_bullet
  local model_root=$RUN_BASE/$prompt/$MODEL
  local config=$ROOT/configs/realistic_niah_v6_${prompt}.json
  local mechanism=$model_root/freeze/mechanism_frozen_confirmation.json
  local generations=$model_root/generation/generations.jsonl
  local registry=$model_root/replacement/$phase/selected_cells.jsonl
  local freeze=$model_root/freeze/confirmation_freeze.json
  local selection=$model_root/causal/targeted_retrieval/discovery_formal/analysis/selection.json
  local panel_root=$model_root/causal/targeted_retrieval/${phase}_formal/final_transition_panel
  local anchors=$panel_root/mode_panel.jsonl
  local targeted=$panel_root/targeted_registry.jsonl
  local bank_plan=$model_root/causal/specialized/discovery_formal/bank_plan/retrieval_anchor_bank_plan.csv
  local role expected output
  role=$(phase_role "$phase")
  expected=$(phase_count "$phase")
  output=$model_root/causal/mechanism_diagnostic_extension/decode_aligned_carrier/$phase
  for path in "$config" "$mechanism" "$generations" "$registry" "$selection" \
    "$anchors" "$targeted" "$bank_plan"; do
    [[ -s "$path" ]] || { echo "missing decode-aligned carrier input: $path" >&2; exit 4; }
  done
  local wrapper=(
    "$PYTHON" scripts/run_realistic_niah_v6_kernel.py
    --target targeted-counter-write --v6-config "$config" --phase "$phase"
    --cohort-registry "$registry" --bank-selection "$selection"
  )
  if [[ "$phase" == confirmation ]]; then
    [[ -s "$freeze" ]] || { echo "missing confirmation freeze: $freeze" >&2; exit 4; }
    wrapper+=(--confirmation-freeze "$freeze")
  fi
  wrapper+=(--)
  if [[ -s "$output/analysis/claim_gates.json" ]]; then
    echo "[$(date --iso-8601=seconds)] REUSE bullet_${phase}_decode_aligned_carrier"
    return
  fi
  if manifest_has_shards "$output/trials/manifest.json" "$expected"; then
    echo "[$(date --iso-8601=seconds)] REUSE bullet_${phase}_decode_aligned_carrier trials"
  else
    run_logged "bullet_${phase}_decode_aligned_carrier" \
      "${wrapper[@]}" \
      --mechanism-config "$mechanism" --model "$MODEL" --cache-dir "$CACHE" \
      --device-map auto --torch-dtype bfloat16 --attention-backend sdpa \
      --generations "$generations" --seed-role "$role" \
      --anchor-registry "$anchors" --targeted-registry "$targeted" \
      --bank-plan "$bank_plan" --source-layer "$SOURCE_LAYER" \
      --head-ablation-scope query_through_carrier --resume \
      --output "$output/trials"
  fi
  run_logged "bullet_${phase}_decode_aligned_carrier_analysis" \
    "$PYTHON" scripts/analyze_realistic_niah_v6_targeted_counter_write_diagnostic.py \
    --input "$output/trials" --phase "$phase" --expected-seeds "$expected" \
    --expected-scope query_through_carrier --output "$output/analysis"
}

for mode in index bullet; do
  for phase in discovery confirmation; do
    run_targeted_likelihood "$mode" "$phase"
  done
done

for phase in discovery confirmation; do
  run_decode_aligned_carrier "$phase"
  run_local_terminal "$phase"
done

printf 'PASS\n' >"$GLOBAL_ROOT/$MODEL.COMPLETE"
echo "[$(date --iso-8601=seconds)] COMPLETE $MODEL mechanism diagnostics"
