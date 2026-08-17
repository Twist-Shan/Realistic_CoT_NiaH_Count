#!/usr/bin/env bash
set -euo pipefail

MODEL=${1:?model label required}
GPU=${2:?gpu index required}
FS=/home/ubuntu/CoT-Native-thinking-v5
REPO=$FS/code/Realistic_CoT_NiaH_Count
PY=$FS/venv/bin/python
RUN=$FS/runs/v5_native_thinking_representation
GEN=$RUN/$MODEL/generations.jsonl
OUT=$RUN/$MODEL/answer_query_extension_v3/smoke
CACHE=$FS/cache/huggingface

mkdir -p "$OUT"
cd "$REPO"
printf 'running\n' > "$OUT/status"

CUDA_VISIBLE_DEVICES="$GPU" "$PY" scripts/run_realistic_niah_v5.py \
  attention-answer-query \
  --model "$MODEL" --generations "$GEN" \
  --output "$OUT/attention_discovery_one.csv" \
  --site-id answer_query_v3 --cohort one_to_one_correct \
  --split discovery --max-rows 1 --overwrite \
  --cache-dir "$CACHE" --device-map auto --torch-dtype bfloat16 \
  --attention-backend sdpa

"$PY" scripts/run_realistic_niah_v5.py answer-query-causal-plan \
  --attention "$OUT/attention_discovery_one.csv" --output "$OUT/plan"

mapfile -t PLAN_ROWS < <("$PY" - "$OUT/plan/answer_query_causal_plan.csv" <<'PY'
import sys
import pandas as pd

frame = pd.read_csv(sys.argv[1])
bank_sizes = sorted(set(frame["bank_size"].astype(int)))
smoke_sizes = {bank_sizes[0], bank_sizes[-1]}
chosen = frame.loc[
    frame["bank_size"].isin(smoke_sizes)
    & (
        frame["condition"].astype(str).str.endswith("_ranked")
        | (
            frame["condition"].astype(str).str.endswith(
                "_layer_matched_random"
            )
            & frame["repeat"].eq(1)
        )
    )
]
if set(chosen["mechanism"].astype(str)) != {
    "answer_prompt_aggregation",
    "answer_trace_aggregation",
}:
    raise SystemExit("smoke plan omitted an answer aggregation mechanism")
for index in chosen.index:
    print(int(index))
PY
)

CUDA_VISIBLE_DEVICES="$GPU" "$PY" scripts/run_realistic_niah_v5.py \
  causal-answer-query-heads \
  --model "$MODEL" --generations "$GEN" \
  --plan "$OUT/plan/answer_query_causal_plan.csv" \
  --plan-rows "${PLAN_ROWS[@]}" \
  --output "$OUT/trials_confirmation_one.jsonl" \
  --site-id answer_query_v3 --cohort one_to_one_correct \
  --split confirmation --max-rows 1 --overwrite \
  --cache-dir "$CACHE" --device-map auto --torch-dtype bfloat16 \
  --attention-backend sdpa

"$PY" - "$OUT/trials_confirmation_one.jsonl" "$OUT/audit.json" <<'PY'
import json
import math
import sys
from pathlib import Path

source, output = map(Path, sys.argv[1:])
rows = [json.loads(line) for line in source.read_text().splitlines() if line]
clean = [row for row in rows if row["condition"] == "clean"]
if len(clean) != 1:
    raise SystemExit(f"expected one clean row, got {len(clean)}")
clean = clean[0]
if not clean["target_first_token_exact"]:
    raise SystemExit("corrected answer query clean target is not rank one")
if clean["target_first_token_probability"] <= 0.01:
    raise SystemExit("corrected answer query clean target probability is too low")
mechanisms = {
    row.get("mechanism") for row in rows if row["condition"] != "clean"
}
if mechanisms != {
    "answer_prompt_aggregation",
    "answer_trace_aggregation",
}:
    raise SystemExit(f"mechanism audit failed: {mechanisms}")
deltas = [
    float(row["target_sequence_log_probability"])
    - float(clean["target_sequence_log_probability"])
    for row in rows
    if row["condition"] != "clean"
]
margin_deltas = [
    float(row["target_first_token_logit_margin"])
    - float(clean["target_first_token_logit_margin"])
    for row in rows
    if row["condition"] != "clean"
]
if not all(math.isfinite(value) for value in deltas):
    raise SystemExit("non-finite head intervention delta")
if not all(math.isfinite(value) for value in margin_deltas):
    raise SystemExit("non-finite count-logit margin delta")
if not any(abs(value) > 0.0 for value in margin_deltas):
    raise SystemExit("all answer-query count-logit margin deltas are zero")
audit = {
    "status": "PASS",
    "site_id": "answer_query_v3",
    "clean_target_first_token_probability": clean[
        "target_first_token_probability"
    ],
    "clean_target_first_token_rank": clean["target_first_token_rank"],
    "conditions": len(rows),
    "mechanisms": sorted(mechanisms),
    "nonzero_log_probability_delta": any(abs(value) > 0.0 for value in deltas),
    "log_probability_deltas": deltas,
    "nonzero_count_logit_margin_delta": any(
        abs(value) > 0.0 for value in margin_deltas
    ),
    "count_logit_margin_deltas": margin_deltas,
}
output.write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n")
print(json.dumps(audit, sort_keys=True))
PY

printf 'complete\n' > "$OUT/status"
