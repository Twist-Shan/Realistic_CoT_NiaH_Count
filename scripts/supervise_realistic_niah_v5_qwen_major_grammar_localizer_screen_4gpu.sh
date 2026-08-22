#!/usr/bin/env bash
set -euo pipefail

CODE_ROOT="${CODE_ROOT:-/home/ubuntu/CoT-Native-thinking-v5/code/Realistic_CoT_NiaH_Count}"
RUN_ROOT="${RUN_ROOT:-/home/ubuntu/CoT-Native-thinking-v5/runs/v5_native_major_grammar_localizer_screen_20260821/Qwen3-8B}"
PYTHON="${PYTHON:-/home/ubuntu/CoT-Native-thinking-v5/venv/bin/python}"
CACHE_DIR="${CACHE_DIR:-/home/ubuntu/CoT-Native-thinking-v5/cache/huggingface}"
CONFIG="$CODE_ROOT/configs/realistic_niah_v5.json"
GENERATIONS="$CODE_ROOT/work/v5_trace_parser_v2/Qwen3-8B_generations_reparsed.jsonl"
ROUTING="$CODE_ROOT/configs/realistic_niah_v5_qwen_hybrid_localizer_p0_ablation_routes_frozen.json"
REGISTRY_ROOT="/home/ubuntu/CoT-Native-thinking-v5/runs/v5_native_grammar_specific_p0_20260820/Qwen3-8B"
REGISTRY_ADJ="$REGISTRY_ROOT/head_behavior_adjacent_rank_after_city_p0_k128_fullpanel_pergrammarregistry_v2_v1/selected_anchor_registry.jsonl"
REGISTRY_SAME="$REGISTRY_ROOT/head_behavior_same_unit_rank_before_city_p0_k128_fullpanel_pergrammarregistry_v2_v1/selected_anchor_registry.jsonl"
P0_SOURCE="/home/ubuntu/CoT-Native-thinking-v5/runs/v5_native_unified_p0_20260820/Qwen3-8B/source_attention_p0_all_local_grammars_full_discovery_v2"
P2_SOURCE="/home/ubuntu/CoT-Native-thinking-v5/runs/v5_native_hybrid_supplement_20260820/Qwen3-8B/source_writes/post_marker"
LOG="$RUN_ROOT/logs/supervisor.log"

mkdir -p "$RUN_ROOT/logs" "$RUN_ROOT/locks" "$RUN_ROOT/source_writes" \
  "$RUN_ROOT/plans" "$RUN_ROOT/behavior_v3" "$CACHE_DIR"
exec 9>"$RUN_ROOT/locks/supervisor.lock"
if ! flock -n 9; then
  echo "another Qwen major-grammar localizer screen owns the lock" >&2
  exit 75
fi
exec > >(tee -a "$LOG") 2>&1
cd "$CODE_ROOT"

test -f "$REGISTRY_ADJ"
test -f "$REGISTRY_SAME"
test "$(wc -l < "$REGISTRY_ADJ")" -eq 131
test "$(wc -l < "$REGISTRY_SAME")" -eq 35
test -f "$P0_SOURCE/manifest.json"
test -f "$P2_SOURCE/manifest.json"

source_output() {
  echo "$RUN_ROOT/source_writes/$1"
}

run_source() {
  local gpu="$1" label="$2" role="$3" grammar="$4"
  local output log expected completed
  output="$(source_output "$label")"
  log="$RUN_ROOT/logs/source_${label}.log"
  if test -f "$output/manifest.json"; then
    expected="$($PYTHON -c 'import json,sys; d=json.load(open(sys.argv[1])); print(d.get("eligible_anchor_tasks",-1))' "$output/manifest.json")"
    completed="$($PYTHON -c 'import json,sys; d=json.load(open(sys.argv[1])); print(d.get("completed_shards",-2))' "$output/manifest.json")"
    if test "$expected" -gt 0 && test "$expected" -eq "$completed"; then
      echo "SOURCE_REUSE label=$label shards=$completed"
      return
    fi
  fi
  echo "SOURCE_START gpu=$gpu label=$label role=$role grammar=$grammar"
  CUDA_VISIBLE_DEVICES="$gpu" HF_HOME="$CACHE_DIR" TOKENIZERS_PARALLELISM=false \
    "$PYTHON" scripts/run_realistic_niah_v5.py causal-source-writes \
      --config "$CONFIG" \
      --model Qwen3-8B \
      --cache-dir "$CACHE_DIR" \
      --device-map auto \
      --torch-dtype bfloat16 \
      --attention-backend sdpa \
      --generations "$GENERATIONS" \
      --output "$output" \
      --anchor-role "$role" \
      --target-grammar-class "$grammar" \
      --include-secondary >"$log" 2>&1
  echo "SOURCE_COMPLETE gpu=$gpu label=$label"
}

echo "STAGE source_writes utc=$(date -u +%FT%TZ)"
run_source 0 adj_record_clause record_clause_pre_d1 adjacent_rank_after_city & p0=$!
run_source 1 adj_city_pre city_pre_d1 adjacent_rank_after_city & p1=$!
run_source 2 same_pre_marker pre_marker_d1 same_unit_rank_before_city & p2=$!
run_source 3 same_city_pre city_pre_d1 same_unit_rank_before_city & p3=$!
failed=0
for pid in "$p0" "$p1" "$p2" "$p3"; do
  if ! wait "$pid"; then failed=1; fi
done
if test "$failed" -ne 0; then
  echo "SOURCE_GROUP_FAILED" >&2
  exit 1
fi

METRICS=(
  target_source_attention_mass
  target_source_relative_attention_mass
  target_minus_max_wrong_source_attention_mass
  source_specific_ov_write_norm
)

# label|source|selection-role|selection-grammar|behavior-grammar|metric|cross-grammar
CANDIDATES=()
add_metric_family() {
  local prefix="$1" source="$2" role="$3" selection_grammar="$4" behavior_grammar="$5"
  local metric short
  for metric in "${METRICS[@]}"; do
    case "$metric" in
      target_source_attention_mass) short=abs ;;
      target_source_relative_attention_mass) short=relative ;;
      target_minus_max_wrong_source_attention_mass) short=margin ;;
      source_specific_ov_write_norm) short=ovnorm ;;
    esac
    CANDIDATES+=("${prefix}_${short}|${source}|${role}|${selection_grammar}|${behavior_grammar}|${metric}|0")
  done
}

add_metric_family adj_p0 "$P0_SOURCE" p0_item_end adjacent_rank_after_city adjacent_rank_after_city
add_metric_family adj_record "$(source_output adj_record_clause)" record_clause_pre_d1 adjacent_rank_after_city adjacent_rank_after_city
add_metric_family adj_citypre "$(source_output adj_city_pre)" city_pre_d1 adjacent_rank_after_city adjacent_rank_after_city
CANDIDATES+=("adj_shared_adjbefore_p2_abs|$P2_SOURCE|post_marker|adjacent_rank_before_city|adjacent_rank_after_city|target_source_attention_mass|1")
add_metric_family same_p0 "$P0_SOURCE" p0_item_end same_unit_rank_before_city same_unit_rank_before_city
add_metric_family same_p2 "$P2_SOURCE" post_marker same_unit_rank_before_city same_unit_rank_before_city
CANDIDATES+=("same_premarker_abs|$(source_output same_pre_marker)|pre_marker_d1|same_unit_rank_before_city|same_unit_rank_before_city|target_source_attention_mass|0")
CANDIDATES+=("same_citypre_abs|$(source_output same_city_pre)|city_pre_d1|same_unit_rank_before_city|same_unit_rank_before_city|target_source_attention_mass|0")
CANDIDATES+=("same_shared_adjbefore_p2_abs|$P2_SOURCE|post_marker|adjacent_rank_before_city|same_unit_rank_before_city|target_source_attention_mass|1")

build_plan() {
  local spec="$1" label source role selection_grammar behavior_grammar metric cross output log
  IFS='|' read -r label source role selection_grammar behavior_grammar metric cross <<<"$spec"
  output="$RUN_ROOT/plans/$label"
  log="$RUN_ROOT/logs/plan_${label}.log"
  if test -f "$output/retrieval_anchor_bank_plan.csv"; then
    echo "PLAN_REUSE label=$label"
    return
  fi
  mkdir -p "$output"
  echo "PLAN_START label=$label role=$role metric=$metric selection_grammar=$selection_grammar"
  set +e
  "$PYTHON" scripts/run_realistic_niah_v5.py causal-plan \
    --config "$CONFIG" \
    --source-writes "$source" \
    --output "$output" \
    --bank-size 128 \
    --anchor-role "$role" \
    --target-grammar-class "$selection_grammar" \
    --selection-metric "$metric" \
    --selection-eligibility-scope local \
    --selection-aggregation seed_event_mean \
    --development-smoke \
    --selected-only-smoke >"$log" 2>&1
  status=$?
  set -e
  if test "$status" -ne 0; then
    echo "PLAN_SKIP label=$label status=$status log=$log"
    return
  fi
  echo "PLAN_COMPLETE label=$label"
}

echo "STAGE plans utc=$(date -u +%FT%TZ) candidates=${#CANDIDATES[@]}"
build_plan_worker() {
  local worker="$1" index
  for ((index=worker; index<${#CANDIDATES[@]}; index+=4)); do
    build_plan "${CANDIDATES[$index]}"
  done
}
build_plan_worker 0 & b0=$!
build_plan_worker 1 & b1=$!
build_plan_worker 2 & b2=$!
build_plan_worker 3 & b3=$!
failed=0
for pid in "$b0" "$b1" "$b2" "$b3"; do
  if ! wait "$pid"; then failed=1; fi
done
if test "$failed" -ne 0; then
  echo "PLAN_GROUP_FAILED" >&2
  exit 1
fi

behavior_complete() {
  local output="$1"
  test -f "$output/manifest.json" || return 1
  "$PYTHON" -c 'import json,sys; d=json.load(open(sys.argv[1])); raise SystemExit(0 if int(d.get("scheduled_anchor_condition_trials",-1))>0 and int(d.get("scheduled_anchor_condition_trials",-1))==int(d.get("completed_shards",-2)) else 1)' "$output/manifest.json"
}

run_behavior_worker() {
  local gpu="$1" index spec label source role selection_grammar behavior_grammar metric cross plan output log registry
  for ((index=gpu; index<${#CANDIDATES[@]}; index+=4)); do
    spec="${CANDIDATES[$index]}"
    IFS='|' read -r label source role selection_grammar behavior_grammar metric cross <<<"$spec"
    plan="$RUN_ROOT/plans/$label/retrieval_anchor_bank_plan.csv"
    output="$RUN_ROOT/behavior_v3/$label"
    log="$RUN_ROOT/logs/behavior_${label}.log"
    if test "$behavior_grammar" = adjacent_rank_after_city; then
      registry="$REGISTRY_ADJ"
    else
      registry="$REGISTRY_SAME"
    fi
    if ! test -f "$plan"; then
      echo "BEHAVIOR_SKIP_NO_PLAN gpu=$gpu label=$label"
      continue
    fi
    if behavior_complete "$output"; then
      echo "BEHAVIOR_REUSE gpu=$gpu label=$label"
      continue
    fi
    echo "BEHAVIOR_START gpu=$gpu label=$label behavior_grammar=$behavior_grammar"
    command=(
      "$PYTHON" scripts/run_realistic_niah_v5.py causal-heads-behavior
      --config "$CONFIG"
      --model Qwen3-8B
      --cache-dir "$CACHE_DIR"
      --device-map auto
      --torch-dtype bfloat16
      --attention-backend sdpa
      --generations "$GENERATIONS"
      --plan "$plan"
      --output "$output"
      --anchor-routing "$ROUTING"
      --behavior-target-grammar-class "$behavior_grammar"
      --evaluation-split discovery
      --conditions selected_bank
      --include-secondary
      --limit 30
      --anchor-sampling prompt_balanced
      --max-new-tokens 256
      --decode-head-ablation-steps -1
      --allow-selection-intervention-site-decoupling
    )
    if test "$cross" -eq 1; then
      command+=(--allow-cross-grammar-bank-transfer --allow-selection-scope-bank-transfer)
    fi
    CUDA_VISIBLE_DEVICES="$gpu" HF_HOME="$CACHE_DIR" TOKENIZERS_PARALLELISM=false \
      "${command[@]}" >"$log" 2>&1
    echo "BEHAVIOR_COMPLETE gpu=$gpu label=$label"
  done
}

echo "STAGE crossfit_discovery_behavior utc=$(date -u +%FT%TZ)"
run_behavior_worker 0 & w0=$!
run_behavior_worker 1 & w1=$!
run_behavior_worker 2 & w2=$!
run_behavior_worker 3 & w3=$!
failed=0
for pid in "$w0" "$w1" "$w2" "$w3"; do
  if ! wait "$pid"; then failed=1; fi
done
if test "$failed" -ne 0; then
  echo "BEHAVIOR_GROUP_FAILED" >&2
  exit 1
fi

"$PYTHON" - "$RUN_ROOT" <<'PY'
import csv,json,sys
from pathlib import Path
root=Path(sys.argv[1])
rows=[]
for output in sorted((root/'behavior_v3').iterdir()):
    shards=sorted((output/'shards').glob('*.jsonl'))
    records=[]
    for shard in shards:
        records.extend(json.loads(line) for line in shard.read_text().splitlines() if line.strip())
    selected=[r for r in records if r.get('condition')=='selected_bank' and r.get('trial_complete')]
    if not selected: continue
    failures=sum(r.get('behavior_outcome')!='correct_next_needle' for r in selected)
    first=selected[0]
    rows.append({
        'candidate':output.name,
        'behavior_grammar':first.get('behavior_target_grammar_class'),
        'selection_grammar':first.get('head_selection_target_grammar_class'),
        'selection_anchor_role':first.get('head_selection_anchor_role'),
        'n':len(selected),
        'failures':failures,
        'failure_rate':failures/len(selected),
    })
rows.sort(key=lambda r:(r['behavior_grammar'],-r['failure_rate'],r['candidate']))
with (root/'screen_results.csv').open('w',newline='') as f:
    w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
(root/'screen_complete.json').write_text(json.dumps({'status':'PASS','candidate_count':len(rows),'rows':rows},indent=2)+'\n')
for row in rows: print(row)
PY

echo "COMPLETE utc=$(date -u +%FT%TZ)"
