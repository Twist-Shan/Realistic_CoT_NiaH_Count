#!/usr/bin/env bash
set -euo pipefail

CODE_ROOT="${CODE_ROOT:-/home/ubuntu/CoT-Native-thinking-v5/code/Realistic_CoT_NiaH_Count_native_v5_p0_20260820}"
RUN_ROOT="${RUN_ROOT:-/home/ubuntu/CoT-Native-thinking-v5/runs/v5_native_unified_p0_20260820/Qwen3-8B}"
PYTHON="${PYTHON:-/home/ubuntu/CoT-Native-thinking-v5/venv/bin/python}"
CACHE_DIR="${CACHE_DIR:-/home/ubuntu/CoT-Native-thinking-v5/cache/huggingface}"
GENERATIONS="$CODE_ROOT/work/v5_trace_parser_v2/Qwen3-8B_generations_reparsed.jsonl"
CONFIG="$CODE_ROOT/configs/realistic_niah_v5.json"
ROUTING="$CODE_ROOT/configs/realistic_niah_v5_qwen_unified_p0_k128_causal_routes_dev.json"
SOURCE_WRITES="$RUN_ROOT/source_attention_p0_all_local_grammars_full_discovery_v2"
SOURCE_COMPLETE="$RUN_ROOT/qwen_unified_p0_full_discovery_source_complete.json"
REGISTRY="$RUN_ROOT/head_behavior_unified_p0_pooledbank_k128_selected_smoke30_v1/selected_anchor_registry.jsonl"
LOG="$RUN_ROOT/logs/qwen_unified_p0_fullseed_selected_dose_grid_supervisor.log"
COMPLETE="$RUN_ROOT/qwen_unified_p0_fullseed_selected_dose_grid_complete.json"
K_LIST=(32 64 80 96 104 110 112 120 128)

mkdir -p "$RUN_ROOT/logs" "$RUN_ROOT/locks" "$CACHE_DIR"
exec 9>"$RUN_ROOT/locks/qwen_unified_p0_fullseed_selected_dose_grid.lock"
if ! flock -n 9; then
  echo "another fullseed P0 selected dose-grid supervisor owns the lock" >&2
  exit 75
fi

while ! test -f "$SOURCE_COMPLETE"; do
  echo "WAIT source_complete utc=$(date -u +%FT%TZ)" | tee -a "$LOG"
  sleep 30
done

run_logged() {
  local label="$1"
  echo "START $label utc=$(date -u +%FT%TZ)" | tee -a "$LOG"
  shift
  "$@" 2>&1 | tee -a "$LOG"
  echo "COMPLETE $label utc=$(date -u +%FT%TZ)" | tee -a "$LOG"
}

cd "$CODE_ROOT"
for bank_size in "${K_LIST[@]}"; do
  plan="$RUN_ROOT/causal_plan_p0_fullseed_local_k${bank_size}_selectedonly_grid_v2"
  output="$RUN_ROOT/head_behavior_unified_p0_fullseed_local_k${bank_size}_selected_grid30_v2"
  run_logged "plan_k${bank_size}" \
    "$PYTHON" scripts/run_realistic_niah_v5.py causal-plan \
      --config "$CONFIG" \
      --source-writes "$SOURCE_WRITES" \
      --output "$plan" \
      --bank-size "$bank_size" \
      --anchor-role p0_item_end \
      --selection-metric target_source_attention_mass \
      --selection-eligibility-scope local \
      --selection-aggregation seed_event_mean \
      --full-panel-plan \
      --selected-only-smoke
  run_logged "behavior_k${bank_size}" \
    env HF_HOME="$CACHE_DIR" TOKENIZERS_PARALLELISM=false \
    "$PYTHON" scripts/run_realistic_niah_v5.py causal-heads-behavior \
      --config "$CONFIG" \
      --model Qwen3-8B \
      --cache-dir "$CACHE_DIR" \
      --device-map auto \
      --torch-dtype bfloat16 \
      --attention-backend sdpa \
      --generations "$GENERATIONS" \
      --plan "$plan/retrieval_anchor_bank_plan.csv" \
      --output "$output" \
      --anchor-routing "$ROUTING" \
      --behavior-all-routed-grammars \
      --allow-selection-scope-bank-transfer \
      --evaluation-split discovery \
      --conditions selected_bank \
      --include-secondary \
      --limit 30 \
      --anchor-sampling prompt_balanced \
      --anchor-registry-input "$REGISTRY" \
      --max-new-tokens 256 \
      --decode-head-ablation-steps -1
done

"$PYTHON" - "$RUN_ROOT" "$COMPLETE" <<'PY'
import datetime
import hashlib
import json
import pathlib
import sys

run_root = pathlib.Path(sys.argv[1])
complete = pathlib.Path(sys.argv[2])
doses = [32, 64, 80, 96, 104, 110, 112, 120, 128]
summary = []
for bank_size in doses:
    output = run_root / (
        f"head_behavior_unified_p0_fullseed_local_k{bank_size}_selected_grid30_v2"
    )
    rows = []
    for path in sorted((output / "shards").glob("trial_*.jsonl")):
        with path.open(encoding="utf-8") as handle:
            rows.extend(json.loads(line) for line in handle if line.strip())
    assert len(rows) == 30, (bank_size, len(rows))
    assert {row["condition"] for row in rows} == {"selected_bank"}
    failures = sum(
        str(row.get("behavior_outcome")) != "correct_next_needle"
        for row in rows
    )
    summary.append(
        {
            "bank_size": bank_size,
            "selected_failures": failures,
            "selected_failure_rate": failures / len(rows),
            "output": str(output),
        }
    )
registry = run_root / (
    "head_behavior_unified_p0_pooledbank_k128_selected_smoke30_v1/"
    "selected_anchor_registry.jsonl"
)
payload = {
    "schema_version": "realistic_niah_v5_qwen_unified_p0_fullseed_dose_grid_v1",
    "status": "PASS",
    "completed_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
    "selection_anchor_role": "p0_item_end",
    "selection_eligibility_scope": "local",
    "selection_aggregation": "equal_seed_mean_of_within_seed_event_means",
    "selection_metric": "target_source_attention_mass",
    "discovery_seed_count": 20,
    "anchor_count": 30,
    "registry_sha256": hashlib.sha256(registry.read_bytes()).hexdigest(),
    "doses": summary,
}
complete.write_text(
    json.dumps(payload, indent=2, sort_keys=True) + "\n",
    encoding="utf-8",
)
print(json.dumps(payload, sort_keys=True))
PY

echo "ALL_COMPLETE utc=$(date -u +%FT%TZ)" | tee -a "$LOG"
