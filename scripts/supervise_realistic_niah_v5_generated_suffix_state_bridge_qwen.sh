#!/usr/bin/env bash
set -euo pipefail

GPU_INDEX=${1:?usage: $0 GPU_INDEX PANEL_ID GEOMETRIES_CSV}
PANEL_ID=${2:?usage: $0 GPU_INDEX PANEL_ID GEOMETRIES_CSV}
GEOMETRIES_CSV=${3:?usage: $0 GPU_INDEX PANEL_ID GEOMETRIES_CSV}
ROOT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$ROOT_DIR"

MODEL=Qwen3-8B
PYTHON="$ROOT_DIR/.venv/bin/python"
RUNNER="$ROOT_DIR/scripts/run_realistic_niah_v5_generated_suffix_state_bridge.py"
ANALYZER="$ROOT_DIR/scripts/analyze_realistic_niah_v5_generated_suffix_state_bridge.py"
DEV_CONFIG="$ROOT_DIR/configs/realistic_niah_v5_native_count_stream_dev.json"
CONFIRM_CONFIG="$ROOT_DIR/configs/realistic_niah_v5_native_count_stream_confirmation_v1.json"
V5_CONFIG="$ROOT_DIR/configs/realistic_niah_v5.json"
GENERATIONS="$ROOT_DIR/work/v5_trace_parser_v2/${MODEL}_generations_reparsed.jsonl"
TARGETED_ROOT="$ROOT_DIR/work/v5_native_count_stream/targeted_count_chain_k128_20d10c_20260821_v1/$MODEL"
BANK_PLAN="$TARGETED_ROOT/frozen_targeted_count_plan.csv"
TARGETED_REGISTRY="$TARGETED_ROOT/final_transition_registry/selected_anchor_registry.jsonl"
PROTOCOL_ROOT="$ROOT_DIR/work/v5_native_count_stream/qwen_generated_suffix_state_balanced_protocol_20d10c_20260821_v1"
OUTPUT_ROOT="$ROOT_DIR/work/v5_native_count_stream/qwen_generated_suffix_state_balanced_20d10c_20260821_v1/$MODEL/$PANEL_ID"

case "$PANEL_ID" in
  count2_6_balanced)
    ANCHORS="$PROTOCOL_ROOT/qwen_count2_6_balanced_anchor_registry.jsonl"
    ANCHOR_MANIFEST="$PROTOCOL_ROOT/qwen_count2_6_balanced_anchor_manifest.json"
    EXPECTED_COUNTS="2,3,4,5,6"
    ;;
  count9_10_balanced)
    ANCHORS="$PROTOCOL_ROOT/qwen_count9_10_balanced_anchor_registry.jsonl"
    ANCHOR_MANIFEST="$PROTOCOL_ROOT/qwen_count9_10_balanced_anchor_manifest.json"
    EXPECTED_COUNTS="9,10"
    ;;
  *)
    echo "unknown Qwen generated-suffix panel: $PANEL_ID" >&2
    exit 2
    ;;
esac

mkdir -p "$OUTPUT_ROOT/logs" "$OUTPUT_ROOT/locks"
exec 9>"$OUTPUT_ROOT/locks/supervisor.lock"
if ! flock -n 9; then
  echo "another Qwen generated-suffix supervisor owns $PANEL_ID" >&2
  exit 3
fi

"$PYTHON" - "$BANK_PLAN" "$ANCHORS" "$ANCHOR_MANIFEST" "$TARGETED_REGISTRY" "$EXPECTED_COUNTS" <<'PY'
import collections, hashlib, json, pathlib, pandas as pd, sys
bank, anchors, manifest, targeted = map(pathlib.Path, sys.argv[1:5])
counts = tuple(int(value) for value in sys.argv[5].split(','))
frame = pd.read_csv(bank)
assert "selection_rank" not in frame.columns
frame = frame[frame["model_label"].astype(str).eq("Qwen3-8B")]
assert frame["condition"].astype(str).value_counts().to_dict() == {
    "layer_matched_random": 3, "selected_bank": 1
}
selected = frame[frame["condition"].astype(str).eq("selected_bank")]
assert len(selected) == 1 and int(selected.iloc[0]["bank_size"]) == 128
assert str(selected.iloc[0]["bank_sha256"]) == "ef30a8a083468c6e88cb5b0924403884ad758fedbc743de36dd03ab9bc4a742b"
rows = [json.loads(line) for line in anchors.read_text().splitlines() if line.strip()]
plan = json.loads(manifest.read_text())
targeted_rows = [json.loads(line) for line in targeted.read_text().splitlines() if line.strip()]
assert len(rows) == 30 and len({int(row["seed"]) for row in rows}) == 30
assert plan["outcome_blind"] is True and plan["selection_rank_used"] is False
assert sorted(plan["counts"]) == sorted(counts)
assert all(row["outcome_blind"] is True for row in rows)
assert not any("selection_rank" in row for row in rows + targeted_rows)
assert {row["request_id"] for row in rows} <= {
    row["request_id"] for row in targeted_rows
}
for phase, seeds, quota in (
    ("discovery", range(1234, 1254), 20 // len(counts)),
    ("confirmation", range(1254, 1264), 10 // len(counts)),
):
    active = [row for row in rows if row["selection_phase"] == phase]
    assert sorted(int(row["seed"]) for row in active) == list(seeds)
    assert collections.Counter(int(row["gold_count"]) for row in active) == {
        count: quota for count in counts
    }
print(json.dumps({
    "status": "PASS",
    "panel_id": plan["panel_id"],
    "plan_sha256": plan["plan_sha256"],
    "anchor_registry_sha256": hashlib.sha256(anchors.read_bytes()).hexdigest(),
}, sort_keys=True))
PY

timestamp() { date -u +%Y-%m-%dT%H:%M:%SZ; }
run_phase() {
  local geometry=$1
  local role=$2
  local phase=$3
  local mechanism=$4
  local root="$OUTPUT_ROOT/$geometry/$phase"
  local analysis="$OUTPUT_ROOT/$geometry/analysis_${phase}"
  local log="$OUTPUT_ROOT/logs/${geometry}.log"
  echo "START geometry=$geometry phase=$phase utc=$(timestamp)" | tee -a "$log"
  CUDA_VISIBLE_DEVICES="$GPU_INDEX" "$PYTHON" "$RUNNER" \
    --mechanism-config "$mechanism" \
    --v5-config "$V5_CONFIG" \
    --model "$MODEL" \
    --cache-dir "$ROOT_DIR/work/hf_cache" \
    --device-map auto \
    --torch-dtype bfloat16 \
    --attention-backend sdpa \
    --generations "$GENERATIONS" \
    --seed-role "$role" \
    --anchor-registry "$ANCHORS" \
    --targeted-registry "$TARGETED_REGISTRY" \
    --bank-plan "$BANK_PLAN" \
    --source-layer 19 \
    --state-patch-geometry "$geometry" \
    --selection-rule deterministic_exact_count_quota_matching_using_anchor_metadata_only \
    --panel-id "$PANEL_ID" \
    --max-new-tokens 16 \
    --resume \
    --output "$root" 2>&1 | tee -a "$log"
  "$PYTHON" "$ANALYZER" \
    --input "$root" \
    --phase "$phase" \
    --bootstrap-samples 10000 \
    --random-seed 20260821 \
    --output "$analysis" 2>&1 | tee -a "$log"
  echo "SEALED geometry=$geometry phase=$phase utc=$(timestamp)" | tee -a "$log"
}

IFS=',' read -r -a GEOMETRIES <<< "$GEOMETRIES_CSV"
for geometry in "${GEOMETRIES[@]}"; do
  case "$geometry" in
    terminal_span|generated_suffix_span|terminal_prefix_span) ;;
    *) echo "unknown state geometry: $geometry" >&2; exit 2 ;;
  esac
  run_phase "$geometry" development discovery "$DEV_CONFIG"
  run_phase "$geometry" confirmation confirmation "$CONFIRM_CONFIG"
  "$PYTHON" - "$OUTPUT_ROOT" "$geometry" "$PANEL_ID" <<'PY'
import json, os, pathlib, sys
root, geometry, panel = pathlib.Path(sys.argv[1]), sys.argv[2], sys.argv[3]
d = json.loads((root / geometry / "analysis_discovery/claim_gates.json").read_text())
c = json.loads((root / geometry / "analysis_confirmation/claim_gates.json").read_text())
payload = {
    "schema_version": "realistic_niah_v5_qwen_generated_suffix_geometry_complete_v1",
    "status": "PASS",
    "model_label": "Qwen3-8B",
    "panel_id": panel,
    "state_patch_geometry": geometry,
    "discovery_seed_count": d["seed_count"],
    "confirmation_seed_count": c["seed_count"],
    "discovery_bridge_pass": d["generated_suffix_state_bridge_pass"],
    "confirmation_bridge_pass": c["generated_suffix_state_bridge_pass"],
    "complete_bridge_pass": bool(
        d["generated_suffix_state_bridge_pass"]
        and c["generated_suffix_state_bridge_pass"]
    ),
}
path = root / geometry / "generated_suffix_state_complete.json"
tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
tmp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
tmp.replace(path)
print(json.dumps(payload, sort_keys=True))
PY
done

"$PYTHON" - "$OUTPUT_ROOT" "$PANEL_ID" "$GEOMETRIES_CSV" <<'PY'
import json, os, pathlib, sys
root, panel = pathlib.Path(sys.argv[1]), sys.argv[2]
geometries = sys.argv[3].split(',')
completed = {
    geometry: json.loads(
        (root / geometry / "generated_suffix_state_complete.json").read_text()
    )
    for geometry in geometries
}
payload = {
    "schema_version": "realistic_niah_v5_qwen_generated_suffix_panel_complete_v1",
    "status": "PASS",
    "model_label": "Qwen3-8B",
    "panel_id": panel,
    "geometries": completed,
    "any_complete_bridge_pass": any(
        row["complete_bridge_pass"] for row in completed.values()
    ),
}
path = root / "panel_complete.json"
tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
tmp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
tmp.replace(path)
print(json.dumps(payload, sort_keys=True))
PY
