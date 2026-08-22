#!/usr/bin/env bash
set -euo pipefail

MODEL="${1:-Qwen3-8B}"
if [[ "$MODEL" != "Qwen3-8B" ]]; then
  echo "This frozen geometry panel is registered only for Qwen3-8B" >&2
  exit 2
fi

REPO="/home/ubuntu/Realistic_CoT_NiaH_Count"
PY="$REPO/.venv/bin/python"
RUNNER="$REPO/scripts/run_realistic_niah_v5_count_stream.py"
ANALYZER="$REPO/scripts/analyze_realistic_niah_v5_query_mediation.py"
ROOT="$REPO/work/v5_native_count_stream/query_mediation_geometry_panel_capped_k128_20d10c_20260821_v2/$MODEL"
NATIVE_ROOT="$REPO/work/v5_native_count_stream/native_loop_chain_k128_k6_20d10c_20260821_v1/$MODEL"
DISCOVERY_PLAN="$NATIVE_ROOT/plan_discovery_offsets123/native_loop_plan.csv"
CONFIRMATION_PLAN="$NATIVE_ROOT/plan_confirmation_offsets123/native_loop_plan.csv"
HEAD_PLAN="$REPO/work/v5_native_count_stream/query_mediation_ladder_k128_20d10c_20260821_v1/$MODEL/head_plan/query_mediation_head_plan.json"
BASIS="$REPO/work/v5_native_count_stream/representation_20260820/$MODEL/item_end_discovery_basis.npz"
SELECTION="$REPO/configs/realistic_niah_v5_qwen_shared_k128_targeted_selection_frozen.json"
ROUTING="$REPO/configs/realistic_niah_v5_qwen_shared_k128_causal_routes_frozen.json"
GENERATIONS="$REPO/work/v5_trace_parser_v2/Qwen3-8B_generations_reparsed.jsonl"
ENDPOINT_COMPLETE="$REPO/work/v5_native_count_stream/query_mediation_ladder_k128_20d10c_20260821_v1/$MODEL/query_mediation_complete.json"
MECHANISM="$REPO/configs/realistic_niah_v5_native_count_stream_confirmation_v1.json"
V5="$REPO/configs/realistic_niah_v5.json"
LOCK="$ROOT/query_mediation_geometry_panel.lock"

mkdir -p "$ROOT/logs"
exec 9>"$LOCK"
if ! flock -n 9; then
  echo "query-mediation geometry-panel lock conflict: $LOCK" >&2
  exit 3
fi

for required in "$DISCOVERY_PLAN" "$CONFIRMATION_PLAN" "$HEAD_PLAN" "$BASIS" "$SELECTION" "$ROUTING" "$GENERATIONS" "$ENDPOINT_COMPLETE"; do
  if [[ ! -f "$required" ]]; then
    echo "missing frozen query-mediation geometry-panel input: $required" >&2
    exit 4
  fi
done

"$PY" - "$DISCOVERY_PLAN" "$CONFIRMATION_PLAN" "$HEAD_PLAN" "$ENDPOINT_COMPLETE" <<'PY'
import json
import pathlib
import sys

import pandas as pd

discovery_path, confirmation_path, head_path, endpoint_path = map(pathlib.Path, sys.argv[1:])
head = json.loads(head_path.read_text(encoding="utf-8"))
endpoint = json.loads(endpoint_path.read_text(encoding="utf-8"))
assert head["model_label"] == "Qwen3-8B"
assert head["targeted_bank_sha256"] == "ef30a8a083468c6e88cb5b0924403884ad758fedbc743de36dd03ab9bc4a742b"
assert head["active_selected_sha256"] == "571e3a3c8d82ca8533def2bab1133f92e6ded5215b6cb1f479ad4ca256ac3ff6"
assert head["layer_matched_random_sha256"] == "e38ea2f6fb8b7cdf3edd32b0f457d22f5b545e926fb17ecd4546318507f709df"
assert head["random_control_overlap_count"] == 0
assert head["selection_rank_used"] is False and head["outcome_blind"] is True
assert endpoint["model_label"] == "Qwen3-8B" and endpoint["selected_geometry"] == "endpoint"

for path, expected_role, expected_seeds in (
    (discovery_path, "development", list(range(1234, 1254))),
    (confirmation_path, "confirmation", list(range(1254, 1264))),
):
    frame = pd.read_csv(path)
    assert "selection_rank" not in frame.columns
    assert set(frame["seed_role"].astype(str)) == {expected_role}
    assert sorted(frame["seed"].astype(int).unique().tolist()) == expected_seeds
    assert not frame["selection_rank_used"].astype(str).str.lower().isin({"true", "1", "yes"}).any()
print("query-mediation geometry-panel contract PASS")
PY

export CUDA_VISIBLE_DEVICES=0
export TOKENIZERS_PARALLELISM=false

run_trials() {
  local phase="$1"
  local seed_role="$2"
  local plan="$3"
  local geometry="$4"
  local output="$ROOT/${geometry}_${phase}"
  "$PY" "$RUNNER" p0-query-mediation \
    --mechanism-config "$MECHANISM" \
    --v5-config "$V5" \
    --model "$MODEL" \
    --cache-dir "$REPO/work/hf_cache" \
    --device-map auto \
    --torch-dtype bfloat16 \
    --attention-backend sdpa \
    --generations "$GENERATIONS" \
    --cohort parser_hit \
    --seed-role "$seed_role" \
    --plan "$plan" \
    --basis "$BASIS" \
    --layer 19 \
    --geometry "$geometry" \
    --targeted-selection "$SELECTION" \
    --anchor-routing "$ROUTING" \
    --head-plan "$HEAD_PLAN" \
    --donor-offsets -3 -2 -1 1 2 3 \
    --random-seed 20260821 \
    --skip-greedy \
    --output "$output"
  "$PY" "$ANALYZER" \
    --trials "$output" \
    --phase "$phase" \
    --geometry "$geometry" \
    --bootstrap-samples 10000 \
    --random-seed 20260821 \
    --output "$ROOT/analysis_${geometry}_${phase}"
}

# Both capped geometries and both cohorts run unconditionally.  Each pair keeps
# its frozen identity; realized width is min(requested, receiver, donor) and is
# audited per row.  This is not exact full-span patching or geometry selection.
for geometry in suffix_cap4 suffix_cap8; do
  run_trials discovery development "$DISCOVERY_PLAN" "$geometry"
  run_trials confirmation confirmation "$CONFIRMATION_PLAN" "$geometry"
done

"$PY" - "$ROOT" <<'PY'
import datetime
import json
import pathlib
import sys

root = pathlib.Path(sys.argv[1])
panels = {}
for geometry in ("suffix_cap4", "suffix_cap8"):
    panels[geometry] = {}
    for phase in ("discovery", "confirmation"):
        path = root / f"analysis_{geometry}_{phase}" / "claim_gates.json"
        panels[geometry][phase] = json.loads(path.read_text(encoding="utf-8"))
out = {
    "schema_version": "realistic_niah_v5_query_mediation_geometry_panel_v1",
    "model_label": "Qwen3-8B",
    "status": "COMPLETE",
    "purpose": "prospective_effect_size_geometry_characterization_after_endpoint_outcome",
    "geometry_selection_performed": False,
    "confirmation_runs_unconditionally": True,
    "geometries": ["suffix_cap4", "suffix_cap8"],
    "panels": panels,
    "completed_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
}
path = root / "geometry_panel_complete.json"
temporary = path.with_suffix(path.suffix + ".tmp")
temporary.write_text(json.dumps(out, indent=2, sort_keys=True) + "\n", encoding="utf-8")
temporary.replace(path)
PY
