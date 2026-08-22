#!/usr/bin/env bash
set -euo pipefail

CODE_ROOT="${CODE_ROOT:-/home/ubuntu/CoT-Native-thinking-v5/code/Realistic_CoT_NiaH_Count}"
SCREEN_ROOT="${SCREEN_ROOT:-/home/ubuntu/CoT-Native-thinking-v5/runs/v5_native_major_grammar_localizer_screen_20260821/Qwen3-8B}"
RUN_ROOT="${RUN_ROOT:-/home/ubuntu/CoT-Native-thinking-v5/runs/v5_native_major_grammar_final_localizers_20260821/Qwen3-8B}"
PYTHON="${PYTHON:-/home/ubuntu/CoT-Native-thinking-v5/venv/bin/python}"
CACHE_DIR="${CACHE_DIR:-/home/ubuntu/CoT-Native-thinking-v5/cache/huggingface}"
GPU_COUNT="${GPU_COUNT:-4}"
CONFIG="$CODE_ROOT/configs/realistic_niah_v5.json"
GENERATIONS="$CODE_ROOT/work/v5_trace_parser_v2/Qwen3-8B_generations_reparsed.jsonl"
ROUTING="$CODE_ROOT/configs/realistic_niah_v5_qwen_hybrid_localizer_p0_ablation_routes_frozen.json"
REGISTRY_ROOT="/home/ubuntu/CoT-Native-thinking-v5/runs/v5_native_grammar_specific_p0_20260820/Qwen3-8B"
REGISTRY_ADJ="$REGISTRY_ROOT/head_behavior_adjacent_rank_after_city_p0_k128_fullpanel_pergrammarregistry_v2_v1/selected_anchor_registry.jsonl"
REGISTRY_SAME="$REGISTRY_ROOT/head_behavior_same_unit_rank_before_city_p0_k128_fullpanel_pergrammarregistry_v2_v1/selected_anchor_registry.jsonl"
SOURCE_ADJ="$SCREEN_ROOT/source_writes/adj_city_pre"
SOURCE_SAME="$SCREEN_ROOT/source_writes/same_city_pre"
LOG="$RUN_ROOT/logs/supervisor.log"
DOSES=(32 64 80 96 112 128)

mkdir -p "$RUN_ROOT/logs" "$RUN_ROOT/locks" "$RUN_ROOT/plans" "$RUN_ROOT/behavior" "$CACHE_DIR"
exec 9>"$RUN_ROOT/locks/supervisor.lock"
if ! flock -n 9; then
  echo "another Qwen major-grammar final-dose supervisor owns the lock" >&2
  exit 75
fi
exec > >(tee -a "$LOG") 2>&1
cd "$CODE_ROOT"

if test "$GPU_COUNT" -ne 4; then
  echo "This registered queue requires exactly four GPUs" >&2
  exit 64
fi

verify_registry() {
  local path="$1" rows="$2" sha="$3" actual_rows actual_sha
  test -f "$path"
  actual_rows="$(wc -l < "$path" | tr -d ' ')"
  actual_sha="$(sha256sum "$path" | awk '{print $1}')"
  if test "$actual_rows" != "$rows" || test "$actual_sha" != "$sha"; then
    echo "REGISTRY_MISMATCH path=$path rows=$actual_rows sha=$actual_sha" >&2
    exit 1
  fi
  echo "REGISTRY_OK path=$path rows=$actual_rows sha=$actual_sha"
}

verify_registry "$REGISTRY_ADJ" 131 5401836d457251cd7728015f36dff955bd6f9c35e9777c02c7e843bf193b0d1f
verify_registry "$REGISTRY_SAME" 35 4d55d31d480aa0fc2aafb17aab3968ed68322c5aacc3ce3d712cbd707148d228
test -f "$SOURCE_ADJ/manifest.json"
test -f "$SOURCE_SAME/manifest.json"

# grammar|label|source|selection-role|metric|registry|confirmation-anchors|K
JOBS=()
for k in "${DOSES[@]}"; do
  JOBS+=("adjacent_rank_after_city|adj_citypre_ovnorm|$SOURCE_ADJ|city_pre_d1|source_specific_ov_write_norm|$REGISTRY_ADJ|45|$k")
  JOBS+=("same_unit_rank_before_city|same_citypre_abs|$SOURCE_SAME|city_pre_d1|target_source_attention_mass|$REGISTRY_SAME|15|$k")
done

plan_complete() {
  local output="$1" k="$2"
  test -f "$output/retrieval_anchor_bank_plan.csv" || return 1
  test -f "$output/causal_plan_audit.json" || return 1
  grep -q "\"registered_bank_size\": $k" "$output/causal_plan_audit.json"
}

build_plan() {
  local spec="$1" grammar label source role metric registry anchors k output log matching status control
  IFS='|' read -r grammar label source role metric registry anchors k <<<"$spec"
  output="$RUN_ROOT/plans/$label/k$k"
  log="$RUN_ROOT/logs/plan_${label}_k${k}.log"
  mkdir -p "$output"
  if plan_complete "$output" "$k"; then
    echo "PLAN_REUSE grammar=$grammar label=$label K=$k"
    return
  fi
  matching=layer_matched
  if test "$k" -eq 128; then matching=global; fi
  echo "PLAN_START grammar=$grammar label=$label K=$k metric=$metric matching=$matching"
  set +e
  "$PYTHON" scripts/run_realistic_niah_v5.py causal-plan \
    --config "$CONFIG" \
    --source-writes "$source" \
    --output "$output" \
    --bank-size "$k" \
    --anchor-role "$role" \
    --target-grammar-class "$grammar" \
    --selection-metric "$metric" \
    --selection-eligibility-scope local \
    --selection-aggregation seed_event_mean \
    --random-control-matching "$matching" \
    --full-panel-plan >"$log" 2>&1
  status=$?
  set -e
  if test "$status" -ne 0 && test "$matching" = layer_matched; then
    echo "PLAN_FALLBACK grammar=$grammar label=$label K=$k from=layer_matched to=global"
    "$PYTHON" scripts/run_realistic_niah_v5.py causal-plan \
      --config "$CONFIG" \
      --source-writes "$source" \
      --output "$output" \
      --bank-size "$k" \
      --anchor-role "$role" \
      --target-grammar-class "$grammar" \
      --selection-metric "$metric" \
      --selection-eligibility-scope local \
      --selection-aggregation seed_event_mean \
      --random-control-matching global \
      --full-panel-plan >>"$log" 2>&1
    matching=global
  elif test "$status" -ne 0; then
    echo "PLAN_FAILED grammar=$grammar label=$label K=$k log=$log" >&2
    exit "$status"
  fi
  if test "$matching" = layer_matched; then control=layer_matched_random; else control=global_random; fi
  printf '%s\n' "$control" > "$output/control_condition.txt"
  echo "PLAN_COMPLETE grammar=$grammar label=$label K=$k control=$control"
}

echo "STAGE formal_plan_matrix jobs=${#JOBS[@]} utc=$(date -u +%FT%TZ)"
build_plan_worker() {
  local worker="$1" index
  for ((index=worker; index<${#JOBS[@]}; index+=4)); do
    build_plan "${JOBS[$index]}"
  done
}
build_plan_worker 0 & p0=$!
build_plan_worker 1 & p1=$!
build_plan_worker 2 & p2=$!
build_plan_worker 3 & p3=$!
failed=0
for pid in "$p0" "$p1" "$p2" "$p3"; do
  if ! wait "$pid"; then failed=1; fi
done
if test "$failed" -ne 0; then
  echo "PLAN_GROUP_FAILED" >&2
  exit 1
fi

behavior_complete() {
  local output="$1"
  test -f "$output/manifest.json" || return 1
  "$PYTHON" -c 'import json,sys; d=json.load(open(sys.argv[1])); s=int(d.get("scheduled_anchor_condition_trials",-1)); c=int(d.get("completed_shards",-2)); raise SystemExit(0 if s>0 and s==c else 1)' "$output/manifest.json"
}

control_condition() {
  local plan_dir="$1"
  if test -f "$plan_dir/control_condition.txt"; then
    cat "$plan_dir/control_condition.txt"
  elif grep -q 'layer_matched_random' "$plan_dir/retrieval_anchor_bank_plan.csv"; then
    echo layer_matched_random
  else
    echo global_random
  fi
}

run_behavior_worker() {
  local gpu="$1" index spec grammar label source role metric registry anchors k plan_dir output log control
  for ((index=gpu; index<${#JOBS[@]}; index+=4)); do
    spec="${JOBS[$index]}"
    IFS='|' read -r grammar label source role metric registry anchors k <<<"$spec"
    plan_dir="$RUN_ROOT/plans/$label/k$k"
    output="$RUN_ROOT/behavior/$label/k$k"
    log="$RUN_ROOT/logs/behavior_${label}_k${k}.log"
    control="$(control_condition "$plan_dir")"
    if behavior_complete "$output"; then
      echo "BEHAVIOR_REUSE gpu=$gpu grammar=$grammar K=$k"
      continue
    fi
    echo "BEHAVIOR_START gpu=$gpu grammar=$grammar label=$label K=$k confirmation_anchors=$anchors control=$control"
    CUDA_VISIBLE_DEVICES="$gpu" HF_HOME="$CACHE_DIR" TOKENIZERS_PARALLELISM=false \
      "$PYTHON" scripts/run_realistic_niah_v5.py causal-heads-behavior \
        --config "$CONFIG" \
        --model Qwen3-8B \
        --cache-dir "$CACHE_DIR" \
        --device-map auto \
        --torch-dtype bfloat16 \
        --attention-backend sdpa \
        --generations "$GENERATIONS" \
        --plan "$plan_dir/retrieval_anchor_bank_plan.csv" \
        --output "$output" \
        --anchor-routing "$ROUTING" \
        --behavior-target-grammar-class "$grammar" \
        --evaluation-split confirmation \
        --conditions selected_bank "$control" \
        --include-secondary \
        --limit 300 \
        --anchor-sampling prompt_balanced \
        --anchor-registry-input "$registry" \
        --max-new-tokens 256 \
        --decode-head-ablation-steps -1 \
        --allow-selection-intervention-site-decoupling >"$log" 2>&1
    echo "BEHAVIOR_COMPLETE gpu=$gpu grammar=$grammar label=$label K=$k"
  done
}

echo "STAGE frozen_confirmation_dose workers=4 utc=$(date -u +%FT%TZ)"
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
import hashlib
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
rows = []
for output in sorted((root / "behavior").glob("*/k*")):
    manifest = json.loads((output / "manifest.json").read_text())
    if manifest["scheduled_anchor_condition_trials"] != manifest["completed_shards"]:
        raise SystemExit(f"incomplete behavior output: {output}")
    outcomes = []
    for shard in sorted((output / "shards").glob("trial_*.jsonl")):
        outcomes.append(json.loads(shard.read_text()))
    by_condition = {}
    for row in outcomes:
        condition = row["condition"]
        bucket = by_condition.setdefault(condition, {"trials": 0, "failures": 0})
        bucket["trials"] += 1
        bucket["failures"] += row.get("behavior_outcome") != "correct_next_needle"
    rows.append({"output": str(output), "conditions": by_condition})
payload = {
    "status": "PASS",
    "model": "Qwen3-8B",
    "selection": {
        "adjacent_rank_after_city": {"site": "city_pre_d1", "metric": "source_specific_ov_write_norm"},
        "same_unit_rank_before_city": {"site": "city_pre_d1", "metric": "target_source_attention_mass"},
    },
    "intervention_start": "p0_item_end",
    "decode_head_ablation_steps": -1,
    "doses": [32, 64, 80, 96, 112, 128],
    "results": rows,
}
path = root / "qwen_major_grammar_final_dose_complete.json"
path.write_text(json.dumps(payload, indent=2) + "\n")
print(f"FORMAL_COMPLETE path={path} sha256={hashlib.sha256(path.read_bytes()).hexdigest()}")
PY

echo "SUPERVISOR_COMPLETE utc=$(date -u +%FT%TZ)"
