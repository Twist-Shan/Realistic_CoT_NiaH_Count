#!/usr/bin/env bash
set -euo pipefail

if [[ "$#" -ne 2 ]]; then
  echo "usage: $0 <Qwen3-8B|Gemma4-E4B> <gpu-index>" >&2
  exit 2
fi

FS=/home/ubuntu/CoT-Native-thinking-v5
REPO=$FS/code/Realistic_CoT_NiaH_Count
RUN=$FS/runs/v5_native_thinking_representation
MODEL=$1
GPU=$2
MODEL_DIR=$RUN/$MODEL
ATTENTION=$MODEL_DIR/attention/pre_city_token/attention.csv
OUT_DIR=$MODEL_DIR/causal/pre_city_token
PLAN_DIR=$OUT_DIR/plan
PLAN=$PLAN_DIR/causal_plan.csv
TRIAL_DIR=$OUT_DIR/head_tests/trials
ANALYSIS_DIR=$OUT_DIR/head_tests/analysis
LOG_DIR=$MODEL_DIR/logs
STATUS=$LOG_DIR/pre_city_causal_supervisor.status
MANIFEST=$OUT_DIR/head_tests/chunk_manifest.tsv
ANALYSIS_EXCLUSIONS=$OUT_DIR/head_tests/analysis_exclusions.jsonl

case "$MODEL" in
  Qwen3-8B|Gemma4-E4B) ;;
  *) echo "unknown model: $MODEL" >&2; exit 2 ;;
esac

mkdir -p "$PLAN_DIR" "$TRIAL_DIR" "$ANALYSIS_DIR" "$LOG_DIR"
exec >> "$LOG_DIR/pre_city_causal_supervisor.log" 2>&1

fail() {
  rc=$?
  printf 'failed rc=%s utc=%s\n' "$rc" "$(date -u +%FT%TZ)" > "$STATUS"
  exit "$rc"
}
trap fail ERR

cd "$REPO"
export HF_HOME="$FS/cache/huggingface"
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

test -s "$ATTENTION"
test -s "$MODEL_DIR/generations.jsonl"
if [[ ! -s "$PLAN" ]]; then
  printf 'running utc=%s stage=pre-city-causal-plan\n' \
    "$(date -u +%FT%TZ)" > "$STATUS"
  "$FS/venv/bin/python" scripts/run_realistic_niah_v5.py \
    pre-city-causal-plan \
    --attention "$ATTENTION" \
    --output "$PLAN_DIR" \
    > "$LOG_DIR/pre_city_causal_plan.log" 2>&1
fi
test -s "$PLAN"

"$FS/venv/bin/python" - "$PLAN" "$MODEL" "$MANIFEST" <<'PY'
import sys
from pathlib import Path

import pandas as pd

plan_path, model, manifest_path = sys.argv[1:]
plan = pd.read_csv(plan_path)
plan = plan.loc[plan["model_label"].eq(model)].reset_index(drop=True)
required = {
    "mechanism", "query_variant", "condition", "bank_size",
    "target_needle_raw_mass", "target_needle_relative_mass",
    "confirmation_target_needle_raw_mass",
    "confirmation_target_needle_relative_mass",
}
missing = sorted(required - set(plan.columns))
if missing:
    raise SystemExit(f"pre-city causal plan missing columns: {missing}")
if set(plan["mechanism"].astype(str)) != {"targeted_retrieval"}:
    raise SystemExit("pre-city mechanism contract mismatch")
variants = {"pre_city_d1", "pre_city_d2", "pre_city_anchor"}
if set(plan["query_variant"].astype(str)) != variants:
    raise SystemExit("pre-city variant contract mismatch")
registered = {1, 2, 4, 8, 16, 32}
lines = []
for variant in sorted(variants):
    frame = plan.loc[plan["query_variant"].astype(str).eq(variant)]
    ranked = frame.loc[
        frame["condition"].astype(str).eq(
            "pre_city_targeted_retrieval_ranked"
        )
    ]
    if set(ranked["bank_size"].astype(int)) != registered:
        raise SystemExit(f"{variant} ranked treatment K mismatch")
    for bank_size in sorted(registered):
        chunk = frame.loc[frame["bank_size"].astype(int).eq(bank_size)]
        indices = " ".join(str(int(value)) for value in chunk.index)
        has_random = int(
            chunk["condition"].astype(str).eq("layer_matched_random").any()
        )
        lines.append(f"{variant}\t{bank_size}\t{indices}\t{has_random}\n")
Path(manifest_path).write_text("".join(lines), encoding="utf-8")
PY

: > "$ANALYSIS_EXCLUSIONS"

has_ok_rows() {
  "$FS/venv/bin/python" - "$1" <<'PY'
import json
import sys

with open(sys.argv[1], encoding="utf-8") as handle:
    found = any(
        row.get("status") == "ok" and row.get("transition_phase") == "retrieve"
        for row in (json.loads(line) for line in handle if line.strip())
    )
raise SystemExit(0 if found else 1)
PY
}

while IFS=$'\t' read -r variant bank_size indices has_random; do
  trial=$TRIAL_DIR/${variant}_K${bank_size}.jsonl
  chunk_log=$LOG_DIR/pre_city_causal_${variant}_K${bank_size}.log
  printf 'running utc=%s stage=causal-pre-city-heads variant=%s K=%s\n' \
    "$(date -u +%FT%TZ)" "$variant" "$bank_size" > "$STATUS"
  if [[ ! -s "$trial" ]]; then
    # shellcheck disable=SC2086
    CUDA_VISIBLE_DEVICES="$GPU" "$FS/venv/bin/python" \
      scripts/run_realistic_niah_v5.py causal-pre-city-heads \
      --model "$MODEL" \
      --generations "$MODEL_DIR/generations.jsonl" \
      --plan "$PLAN" \
      --plan-rows $indices \
      --query-variant "$variant" \
      --output "$trial" \
      --cohort one_to_one \
      --cache-dir "$FS/cache/huggingface" \
      --device-map auto \
      --torch-dtype bfloat16 \
      --attention-backend sdpa \
      > "$chunk_log" 2>&1
  fi
  test -s "$trial"

  if ! has_ok_rows "$trial"; then
    printf '{"query_variant":"%s","bank_size":%s,"transition_phase":"retrieve","reason":"no_status_ok_rows"}\n' \
      "$variant" "$bank_size" >> "$ANALYSIS_EXCLUSIONS"
    echo "[$MODEL pre-city causal] skip variant=$variant K=$bank_size no-ok-rows"
    continue
  fi
  outcomes=(
    target_mean_token_log_probability
    target_first_token_exact
    target_sequence_teacher_forced_exact
    target_first_token_rank
  )
  for outcome in "${outcomes[@]}"; do
    stem=${variant}_K${bank_size}_${outcome}
    "$FS/venv/bin/python" scripts/run_realistic_niah_v5.py causal-analyze \
      --trials "$trial" \
      --output "$ANALYSIS_DIR/${stem}_vs_clean.csv" \
      --treatment pre_city_targeted_retrieval_ranked \
      --control clean \
      --outcome "$outcome" \
      --mechanism targeted_retrieval \
      --bank-size "$bank_size" \
      --transition-phase retrieve \
      > "$LOG_DIR/analyze_${stem}_vs_clean.log" 2>&1
    if [[ "$has_random" -eq 1 ]]; then
      "$FS/venv/bin/python" scripts/run_realistic_niah_v5.py causal-analyze \
        --trials "$trial" \
        --output "$ANALYSIS_DIR/${stem}_vs_layer_matched_random.csv" \
        --treatment pre_city_targeted_retrieval_ranked \
        --control layer_matched_random \
        --outcome "$outcome" \
        --mechanism targeted_retrieval \
        --bank-size "$bank_size" \
        --transition-phase retrieve \
        > "$LOG_DIR/analyze_${stem}_vs_random.log" 2>&1
    fi
  done
  echo "[$MODEL pre-city causal] complete variant=$variant K=$bank_size"
done < "$MANIFEST"

"$FS/venv/bin/python" - "$OUT_DIR" "$MODEL" <<'PY'
import json
import sys
from pathlib import Path

out = Path(sys.argv[1])
model = sys.argv[2]
files = sorted((out / "head_tests" / "trials").glob("*.jsonl"))
rows = []
for path in files:
    with path.open(encoding="utf-8") as handle:
        rows.extend(json.loads(line) for line in handle if line.strip())
planned = [row for row in rows if row.get("condition") != "clean"]
required = {
    "target_needle_raw_mass", "target_needle_relative_mass",
    "confirmation_target_needle_raw_mass",
    "confirmation_target_needle_relative_mass",
}
missing = {
    field: sum(field not in row for row in planned)
    for field in sorted(required)
}
if any(missing.values()):
    raise SystemExit(f"pre-city exact-span contract failed: {missing}")
audit = {
    "schema_version": "realistic_niah_v5_pre_city_causal_supervisor_v1",
    "model_label": model,
    "mechanism": "targeted_retrieval",
    "query_variants": ["pre_city_d1", "pre_city_d2", "pre_city_anchor"],
    "selection_split": "discovery",
    "evaluation_split": "confirmation",
    "selection_cohort": "one_to_one",
    "variant_specific_head_banks": True,
    "registered_bank_sizes": [1, 2, 4, 8, 16, 32],
    "trial_files": [str(path.resolve()) for path in files],
    "trial_rows": len(rows),
    "ok_rows": sum(row.get("status") == "ok" for row in rows),
    "excluded_rows": sum(row.get("status") != "ok" for row in rows),
    "planned_rows_with_discovery_and_confirmation_exact_span_mass": len(planned),
    "analysis_files": len(list((out / "head_tests" / "analysis").glob("*.csv"))),
    "analysis_exclusions": [
        json.loads(line)
        for line in (out / "head_tests" / "analysis_exclusions.jsonl").read_text(
            encoding="utf-8"
        ).splitlines()
        if line.strip()
    ],
    "broad_aggregation_used": False,
    "unit_of_inference": "seed",
}
(out / "head_tests" / "pre_city_causal_audit.json").write_text(
    json.dumps(audit, indent=2, sort_keys=True) + "\n", encoding="utf-8"
)
PY

printf 'complete utc=%s chunks=18\n' "$(date -u +%FT%TZ)" > "$STATUS"
echo "[$MODEL pre-city causal supervisor] complete"
