#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 2 ]]; then
  echo "usage: $0 <Qwen3-8B|Gemma4-E4B> <gpu-index>" >&2
  exit 2
fi
MODEL=$1
GPU_INDEX=$2
case "$MODEL" in
  Qwen3-8B|Gemma4-E4B) ;;
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
PROTOCOL=$ROOT/configs/realistic_niah_v6_mechanism_followup_v2.json
GLOBAL_ROOT=$RUN_BASE/mechanism_followup_v2
LOG_ROOT=$GLOBAL_ROOT/logs/$MODEL
LOCK_ROOT=$GLOBAL_ROOT/locks
mkdir -p "$LOG_ROOT" "$LOCK_ROOT"
cd "$ROOT"

for path in "$PYTHON" "$PROTOCOL"; do
  [[ -s "$path" ]] || { echo "missing V2 follow-up input: $path" >&2; exit 4; }
done

exec 9>"$LOCK_ROOT/$MODEL.lock"
if ! flock -n 9; then
  echo "another $MODEL V2 follow-up supervisor owns the lock" >&2
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

manifest_has_shards() {
  local manifest=$1
  local expected=$2
  [[ -s "$manifest" ]] || return 1
  "$PYTHON" -c \
    'import json,sys; x=json.load(open(sys.argv[1])); raise SystemExit(0 if int(x.get("completed_shards",-1))==int(sys.argv[2]) else 1)' \
    "$manifest" "$expected"
}

run_index_city_support() {
  local prompt=enumeration_index
  local model_root=$RUN_BASE/$prompt/$MODEL
  local config=$ROOT/configs/realistic_niah_v6_${prompt}.json
  local generations=$model_root/generation/generations.jsonl
  local discovery_registry=$model_root/replacement/discovery/selected_cells.jsonl
  local confirmation_registry=$model_root/replacement/confirmation/selected_cells.jsonl
  local freeze=$model_root/freeze/confirmation_freeze.json
  local selection=$model_root/causal/targeted_retrieval/discovery_formal/analysis/selection.json
  local selected_k plan output
  for path in "$config" "$generations" "$discovery_registry" \
    "$confirmation_registry" "$freeze" "$selection"; do
    [[ -s "$path" ]] || { echo "missing Index support input: $path" >&2; exit 4; }
  done
  selected_k=$($PYTHON -c \
    'import json,sys; print(int(json.load(open(sys.argv[1]))["selected_k"]))' \
    "$selection")
  plan=$model_root/causal/targeted_retrieval/discovery_formal/plans/k$selected_k/retrieval_anchor_bank_plan.csv
  [[ -s "$plan" ]] || { echo "missing frozen Index plan: $plan" >&2; exit 4; }
  output=$GLOBAL_ROOT/index_targeted_city_support/$MODEL
  mkdir -p "$output"
  if ! manifest_has_shards "$output/trials/manifest.json" 50; then
    run_logged index_targeted_city_support \
      "$PYTHON" scripts/run_realistic_niah_v6_causal.py \
      --v6-config "$config" --model-label "$MODEL" --phase confirmation \
      --confirmation-freeze "$freeze" \
      --cohort-registry "$confirmation_registry" \
      --causal-membership-registry "$discovery_registry" -- causal-heads \
      --model "$MODEL" --cache-dir "$CACHE" --device-map auto \
      --torch-dtype bfloat16 --attention-backend sdpa \
      --generations "$generations" --plan "$plan" \
      --head-ablation-scope registered_query_through_city_prefix \
      --output "$output/trials" --limit 10
  else
    echo "[$(date --iso-8601=seconds)] REUSE index_targeted_city_support"
  fi
  run_logged index_targeted_city_support_analysis \
    "$PYTHON" scripts/analyze_realistic_niah_v6_targeted_city_likelihood.py \
    --trials "$output/trials" --phase confirmation --expected-seeds 10 \
    --output "$output/analysis"
  run_logged index_targeted_city_position_audit \
    "$PYTHON" scripts/audit_realistic_niah_v6_targeted_city_positions.py \
    --trials "$output/trials" --expected-seeds 10 \
    --expected-scope registered_query_through_city_prefix \
    --output "$output/position_audit.json"
}

run_full_item_greedy() {
  local prompt=$1
  local direction=$2
  local receiver layer
  case "$direction" in
    forward_skip) receiver=5 ;;
    backward_rewind) receiver=7 ;;
    *) return 2 ;;
  esac
  case "$MODEL:$prompt" in
    Qwen3-8B:enumeration_index|Qwen3-8B:enumeration_bullet) layer=0 ;;
    Gemma4-E4B:enumeration_index|Gemma4-E4B:enumeration_bullet) layer=21 ;;
    *) return 2 ;;
  esac
  local model_root=$RUN_BASE/$prompt/$MODEL
  local config=$ROOT/configs/realistic_niah_v6_${prompt}.json
  local generations=$model_root/generation/generations.jsonl
  local registry=$model_root/replacement/confirmation/selected_cells.jsonl
  local freeze=$model_root/freeze/confirmation_freeze.json
  local output=$GLOBAL_ROOT/full_item_greedy/$prompt/$MODEL/$direction
  for path in "$config" "$generations" "$registry" "$freeze"; do
    [[ -s "$path" ]] || { echo "missing full-item greedy input: $path" >&2; exit 4; }
  done
  if [[ -s "$output/manifest.json" ]] && "$PYTHON" - "$output/manifest.json" "$layer" <<'PY'
import json, sys
x=json.load(open(sys.argv[1]))
ok=(x.get("status")=="PASS" and x.get("patch_scope")=="item_span"
    and list(map(int,x.get("layers",[])))==[int(sys.argv[2])]
    and set(x.get("generation_conditions",[]))=={"receiver_self","native_donor","donor_to_receiver"}
    and int(x.get("seed_count",-1))==10)
raise SystemExit(0 if ok else 1)
PY
  then
    echo "[$(date --iso-8601=seconds)] REUSE full_item_${prompt}_${direction}"
    return
  fi
  run_logged "full_item_${prompt}_${direction}_L${layer}" \
    "$PYTHON" scripts/run_realistic_niah_v6_kernel.py \
    --target natural-aligned-progress --v6-config "$config" \
    --phase confirmation --confirmation-freeze "$freeze" \
    --cohort-registry "$registry" -- \
    --model "$MODEL" --cache-dir "$CACHE" --device-map auto \
    --torch-dtype bfloat16 --attention-backend sdpa \
    --generations "$generations" --gold-count 10 \
    --donor-occurrence 6 --receiver-occurrence "$receiver" --tail-offset 0 \
    --patch-scope item_span --layers "$layer" \
    --conditions receiver_self native_donor donor_to_receiver \
    --generation-conditions receiver_self native_donor donor_to_receiver \
    --max-new-tokens 128 --output "$output"
}

run_fresh_carrier_replication() {
  [[ "$MODEL" == "Gemma4-E4B" ]] || return 0
  local prompt=enumeration_bullet
  local model_root=$RUN_BASE/$prompt/$MODEL
  local config=$ROOT/configs/realistic_niah_v6_${prompt}.json
  local mechanism=$model_root/freeze/mechanism_frozen_confirmation.json
  local freeze=$model_root/freeze/confirmation_freeze.json
  local generations=$model_root/generation/generations.jsonl
  local base_registry=$model_root/replacement/confirmation/selected_cells.jsonl
  local discovery_registry=$model_root/replacement/discovery/selected_cells.jsonl
  local selection=$model_root/causal/targeted_retrieval/discovery_formal/analysis/selection.json
  local bank_plan=$model_root/causal/specialized/discovery_formal/bank_plan/retrieval_anchor_bank_plan.csv
  local root=$GLOBAL_ROOT/fresh_bullet_gemma_carrier
  local cohort_root=$root/cohort
  local registry=$cohort_root/selected_cells.jsonl
  local cohort_lock=$cohort_root/cohort_lock.json
  local source_writes=$root/source_writes/p0_item_end
  local panel=$root/final_transition_panel
  local trials=$root/trials
  local analysis=$root/analysis
  for path in "$config" "$mechanism" "$freeze" "$generations" \
    "$base_registry" "$discovery_registry" "$selection" "$bank_plan"; do
    [[ -s "$path" ]] || { echo "missing fresh carrier input: $path" >&2; exit 4; }
  done
  if [[ ! -s "$cohort_lock" ]]; then
    mapfile -t exclude_registries < <(
      find "$model_root/replacement" -type f -name selected_cells.jsonl | sort
    )
    local freeze_args=(
      "$PYTHON" scripts/freeze_realistic_niah_v6_fresh_carrier_replication_cohort.py
      --v6-config "$config" --protocol "$PROTOCOL"
      --generations "$generations"
      --base-confirmation-registry "$base_registry"
      --model "$MODEL" --gold-count 10 --quota 10 --output "$cohort_root"
    )
    local path
    for path in "${exclude_registries[@]}"; do
      freeze_args+=(--exclude-registry "$path")
    done
    run_logged fresh_carrier_cohort_freeze "${freeze_args[@]}"
  else
    echo "[$(date --iso-8601=seconds)] REUSE fresh_carrier_cohort_lock"
  fi

  if ! manifest_has_shards "$source_writes/manifest.json" 10; then
    run_logged fresh_carrier_source_writes \
      "$PYTHON" scripts/run_realistic_niah_v6_causal.py \
      --v6-config "$config" --model-label "$MODEL" --phase confirmation \
      --confirmation-freeze "$freeze" --cohort-registry "$registry" \
      --causal-membership-registry "$discovery_registry" -- causal-source-writes \
      --model "$MODEL" --cache-dir "$CACHE" --device-map auto \
      --torch-dtype bfloat16 --attention-backend sdpa \
      --generations "$generations" --output "$source_writes" \
      --anchor-role p0_item_end --include-secondary --counts 10 \
      --final-transition-only --limit 10
  else
    echo "[$(date --iso-8601=seconds)] REUSE fresh_carrier_source_writes"
  fi
  if [[ ! -s "$panel/manifest.json" ]]; then
    run_logged fresh_carrier_panel \
      "$PYTHON" scripts/build_realistic_niah_v6_final_transition_panel.py \
      --v6-config "$config" --model "$MODEL" --generations "$generations" \
      --cohort-registry "$registry" --source-writes "$source_writes" \
      --seed-role confirmation --output "$panel"
  else
    echo "[$(date --iso-8601=seconds)] REUSE fresh_carrier_panel"
  fi
  if ! manifest_has_shards "$trials/manifest.json" 10; then
    run_logged fresh_carrier_trials \
      "$PYTHON" scripts/run_realistic_niah_v6_kernel.py \
      --target targeted-counter-write --v6-config "$config" \
      --phase confirmation --confirmation-freeze "$freeze" \
      --cohort-registry "$registry" --bank-selection "$selection" -- \
      --mechanism-config "$mechanism" --model "$MODEL" --cache-dir "$CACHE" \
      --device-map auto --torch-dtype bfloat16 --attention-backend sdpa \
      --generations "$generations" --seed-role confirmation \
      --anchor-registry "$panel/mode_panel.jsonl" \
      --targeted-registry "$panel/targeted_registry.jsonl" \
      --bank-plan "$bank_plan" --source-layer 16 \
      --head-ablation-scope query_through_carrier --resume --output "$trials"
  else
    echo "[$(date --iso-8601=seconds)] REUSE fresh_carrier_trials"
  fi
  run_logged fresh_carrier_analysis \
    "$PYTHON" scripts/analyze_realistic_niah_v6_targeted_counter_write_diagnostic.py \
    --input "$trials" --phase confirmation --expected-seeds 10 \
    --expected-scope query_through_carrier --output "$analysis"
  run_logged fresh_carrier_finalize \
    "$PYTHON" scripts/finalize_realistic_niah_v6_fresh_carrier_replication.py \
    --protocol "$PROTOCOL" --cohort-lock "$cohort_lock" \
    --trials "$trials" --analysis "$analysis" \
    --output "$root/replication_complete.json"
}

try_finalize_global() {
  [[ -s "$GLOBAL_ROOT/Qwen3-8B.COMPLETE" ]] || return 0
  [[ -s "$GLOBAL_ROOT/Gemma4-E4B.COMPLETE" ]] || return 0
  exec 8>"$LOCK_ROOT/global_finalize.lock"
  flock 8
  if [[ ! -s "$GLOBAL_ROOT/full_item_greedy/analysis/claim_gates.json" ]]; then
    run_logged full_item_greedy_analysis \
      "$PYTHON" scripts/analyze_realistic_niah_v6_full_item_greedy.py \
      --root "$GLOBAL_ROOT/full_item_greedy" --protocol "$PROTOCOL" \
      --output "$GLOBAL_ROOT/full_item_greedy/analysis"
  fi
  for path in \
    "$GLOBAL_ROOT/index_targeted_city_support/Qwen3-8B/analysis/claim_gates.json" \
    "$GLOBAL_ROOT/index_targeted_city_support/Gemma4-E4B/analysis/claim_gates.json" \
    "$GLOBAL_ROOT/full_item_greedy/analysis/claim_gates.json" \
    "$GLOBAL_ROOT/fresh_bullet_gemma_carrier/replication_complete.json"; do
    [[ -s "$path" ]] || { echo "missing V2 completion artifact: $path" >&2; exit 4; }
  done
  printf 'PASS\n' >"$GLOBAL_ROOT/FOLLOWUP_V2.COMPLETE"
}

run_index_city_support
run_fresh_carrier_replication
for prompt in enumeration_index enumeration_bullet; do
  for direction in forward_skip backward_rewind; do
    run_full_item_greedy "$prompt" "$direction"
  done
done
printf 'PASS\n' >"$GLOBAL_ROOT/$MODEL.COMPLETE"
try_finalize_global
echo "[$(date --iso-8601=seconds)] COMPLETE $MODEL V2 mechanism follow-up"
