#!/usr/bin/env bash
set -euo pipefail

REPO="${REPO:-/home/ubuntu/v445_top6_followups_20260827/repo}"
PYTHON="${PYTHON:-/lambda/nfs/CoT-Non-thinking-v4/venv/bin/python}"
CAMPAIGN="${CAMPAIGN:-/lambda/nfs/CoT-Non-thinking-v4/runs/nonthinking_v445_8gpu_20260813}"
RUN_ROOT="${RUN_ROOT:-/lambda/nfs/CoT-Non-thinking-v4/runs/nonthinking_v445_gemma_top6_followups_20260827}"
PACKED="${PACKED:-/lambda/nfs/CoT-Non-thinking-v4/runs/v4_4_counter_channel_20260806/packed}"
CACHE_DIR="${CACHE_DIR:-/lambda/nfs/CoT-Non-thinking-v4/hf-cache}"
STIMULI="$CAMPAIGN/dataset/canonical_run_20260731_v4_numeric_presentation_v3_stimuli.jsonl"
RESTORATION="$CAMPAIGN/canonical_merged"
BASIS="$RUN_ROOT/analysis/retrieval_geometry/retrieval_bases.pt"
SPAN_CONFIG="configs/realistic_niah_v4_4_5_span_restoration_canonical_top6_extension.json"
SERIAL_CONFIG="configs/realistic_niah_v4_4_5_serial_mediation_top6_extension.json"
INDUCTION_CONFIG="configs/realistic_niah_v4_4_5_induction_circuit_top6_extension.json"
NOISE_CONFIG="configs/realistic_niah_v4_4_5_noise_factorial_top6_extension.json"
ORIGINAL_INDUCTION="/lambda/nfs/CoT-Non-thinking-v4/runs/nonthinking_v445_exp22_induction_v3_20260814/Gemma4-E4B"
ORIGINAL_NOISE="/lambda/nfs/CoT-Non-thinking-v4/runs/nonthinking_v445_exp23_noise_factorial_v2_20260815/Gemma4-E4B"
SEEDS="1254,1255,1256,1257,1258,1259,1260,1261,1262,1263"
COUNTS="1,2,3,4,5,6,7,8,9,10"
EXPECTED_STIMULUS_SHA="da4dd86142eb8a07f9a7e53497efd3375184c8e68367d4db994370fcb331f090"

mkdir -p "$RUN_ROOT/logs" "$RUN_ROOT/locks" "$RUN_ROOT/provenance"
exec 9>"$RUN_ROOT/locks/coordinator.lock"
if ! flock -n 9; then
  echo "another Gemma Top-6 follow-up coordinator owns the lock" >&2
  exit 75
fi

test -x "$PYTHON"
test -f "$STIMULI"
test -f "$PACKED/layers/Gemma4-E4B__answer_query__L37.npz"
test -d "$CACHE_DIR"
[[ "$(sha256sum "$STIMULI" | awk '{print $1}')" == "$EXPECTED_STIMULUS_SHA" ]]

cd "$REPO"
export PYTHONPATH="$REPO/src${PYTHONPATH:+:$PYTHONPATH}"
export TOKENIZERS_PARALLELISM=false
export CUDA_VISIBLE_DEVICES=0

ORIGINAL_FILES=(
  "$CAMPAIGN/analysis/span_restoration/broad_summary.csv"
  "$CAMPAIGN/analysis/retrieval_geometry/retrieval_bases.pt"
  "/lambda/nfs/CoT-Non-thinking-v4/runs/nonthinking_v445_exp19_serial_mediation_20260814/Gemma4-E4B/detail.jsonl"
  "/lambda/nfs/CoT-Non-thinking-v4/runs/nonthinking_v445_exp19_serial_mediation_20260814/Gemma4-E4B/broad_metrics.jsonl"
  "/lambda/nfs/CoT-Non-thinking-v4/runs/nonthinking_v445_exp19_serial_mediation_20260814/Gemma4-E4B/complete.json"
  "/lambda/nfs/CoT-Non-thinking-v4/runs/nonthinking_v445_exp19_serial_mediation_20260814/Gemma4-E4B/run_provenance.json"
  "$ORIGINAL_INDUCTION/detail.jsonl"
  "$ORIGINAL_INDUCTION/synthetic_rows.jsonl"
  "$ORIGINAL_INDUCTION/complete.json"
  "$ORIGINAL_INDUCTION/run_provenance.json"
  "$ORIGINAL_NOISE/factorial_rows.jsonl"
  "$ORIGINAL_NOISE/outside_context_rows.jsonl"
  "$ORIGINAL_NOISE/complete.json"
  "$ORIGINAL_NOISE/run_provenance.json"
)
PRESERVATION_SNAPSHOT="$RUN_ROOT/provenance/original_outputs_before_final.json"
if [[ ! -f "$PRESERVATION_SNAPSHOT" ]]; then
  preservation_args=()
  for original_file in "${ORIGINAL_FILES[@]}"; do
    preservation_args+=(--path "$original_file")
  done
  "$PYTHON" scripts/audit_realistic_niah_v4_4_5_original_preservation.py \
    --snapshot "${preservation_args[@]}" \
    --output "$PRESERVATION_SNAPSHOT"
fi

phase() {
  printf '[%s] %s\n' "$(date -u +%FT%TZ)" "$1"
}

phase "CPU derivations: Top-6 attention response and retrieval geometry"
mkdir -p "$RUN_ROOT/analysis/attention_response" "$RUN_ROOT/analysis/retrieval_geometry"
"$PYTHON" scripts/analyze_realistic_niah_v4_4_5_topk_attention_response.py \
  --broad-summary "$CAMPAIGN/analysis/span_restoration/broad_summary.csv" \
  --experiment-config "$SPAN_CONFIG" \
  --model Gemma4-E4B \
  --output-dir "$RUN_ROOT/analysis/attention_response" \
  >"$RUN_ROOT/logs/attention_response.log" 2>&1
"$PYTHON" scripts/analyze_realistic_niah_v4_4_5_retrieval_geometry.py \
  --run-root "$RESTORATION" \
  --output-dir "$RUN_ROOT/analysis/retrieval_geometry" \
  --experiment-config "$SPAN_CONFIG" \
  --models Gemma4-E4B \
  --bootstrap-draws 200 \
  --sum-registered-head-writes \
  >"$RUN_ROOT/logs/retrieval_geometry.log" 2>&1
"$PYTHON" - "$BASIS" "$RUN_ROOT/analysis/retrieval_geometry/geometry_audit.json" <<'PY'
import json
import sys
import torch

bases = torch.load(sys.argv[1], map_location="cpu", weights_only=True)
assert set(bases) == {"Gemma4-E4B.L29", "Gemma4-E4B.L35"}
assert all(tuple(payload["components"].shape[:1]) == (3,) for payload in bases.values())
audit = json.load(open(sys.argv[2], encoding="utf-8"))
assert audit["status"] == "PASS"
assert audit["bank_construction"] == "sum_registered_per_head_post_o_writes"
assert audit["rows"] == 600
PY
touch "$RUN_ROOT/analysis/.CPU_DERIVATIONS_COMPLETE"
phase "CPU derivations complete"

run_retrieval() {
  local layer="$1"
  local label="Gemma_L${layer}"
  local output="$RUN_ROOT/retrieval_subspace/$label"
  local analysis="$RUN_ROOT/analysis/retrieval_subspace/$label"
  local log="$RUN_ROOT/logs/retrieval_${label}.log"
  local marker="$output/.RUN_AND_ANALYSIS_COMPLETE"
  if [[ -f "$marker" ]]; then
    phase "retrieval $label already complete"
    return
  fi
  phase "retrieval $label start"
  mkdir -p "$output" "$analysis"
  "$PYTHON" scripts/run_realistic_niah_v4_4_5_retrieval_subspace.py \
    --model Gemma4-E4B \
    --stimuli "$STIMULI" \
    --stimuli-config configs/realistic_niah_v4.json \
    --experiment-config "$SPAN_CONFIG" \
    --basis-file "$BASIS" \
    --retrieval-layer "$layer" \
    --source-patch-layer 8 \
    --seeds "$SEEDS" \
    --counts "$COUNTS" \
    --output-dir "$output" \
    --cache-dir "$CACHE_DIR" \
    --device-map cuda \
    >"$log" 2>&1
  "$PYTHON" scripts/analyze_realistic_niah_v4_4_5_retrieval_subspace.py \
    --run-root "$output" \
    --restoration-root "$RESTORATION" \
    --output-dir "$analysis" \
    --models Gemma4-E4B \
    >>"$log" 2>&1
  "$PYTHON" - "$output/Gemma4-E4B/detail.jsonl" "$analysis/analysis_audit.json" "$layer" <<'PY'
import json
import sys

detail_path, audit_path, layer = sys.argv[1], sys.argv[2], int(sys.argv[3])
rows = [json.loads(line) for line in open(detail_path, encoding="utf-8") if line.strip()]
expected_heads = [4, 2] if layer == 29 else [2, 7, 1, 3]
assert len(rows) == 400
assert len({(row["seed"], row["gold_count"], row["condition"]) for row in rows}) == 400
assert all(row["retrieval_heads"] == expected_heads for row in rows)
assert all(int(row["retrieval_layer"]) == layer for row in rows)
audit = json.load(open(audit_path, encoding="utf-8"))
assert audit["status"] == "PASS"
assert audit["rows"] == 400
PY
  touch "$marker"
  phase "retrieval $label complete"
}

run_serial() {
  local output="$RUN_ROOT/serial_mediation"
  local analysis="$RUN_ROOT/analysis/serial_mediation"
  local log="$RUN_ROOT/logs/serial_mediation.log"
  local marker="$output/Gemma4-E4B/.RUN_AND_ANALYSIS_COMPLETE"
  if [[ -f "$marker" ]]; then
    phase "serial mediation already complete"
    return
  fi
  phase "serial mediation start"
  mkdir -p "$output" "$analysis"
  "$PYTHON" scripts/run_realistic_niah_v4_4_5_serial_mediation.py \
    --model Gemma4-E4B \
    --stimuli "$STIMULI" \
    --stimuli-config configs/realistic_niah_v4_4_5_stimuli.json \
    --experiment-config "$SERIAL_CONFIG" \
    --retrieval-basis "$BASIS" \
    --answer-packed-root "$PACKED" \
    --output-dir "$output" \
    --cache-dir "$CACHE_DIR" \
    --device-map cuda \
    >"$log" 2>&1
  "$PYTHON" scripts/analyze_realistic_niah_v4_4_5_serial_mediation.py \
    --run-roots "$output" \
    --output-dir "$analysis" \
    --experiment-config "$SERIAL_CONFIG" \
    --models Gemma4-E4B \
    --bootstrap-draws 10000 \
    >>"$log" 2>&1
  "$PYTHON" - "$output/Gemma4-E4B/complete.json" "$analysis/analysis_audit.json" <<'PY'
import json
import sys

complete = json.load(open(sys.argv[1], encoding="utf-8"))
audit = json.load(open(sys.argv[2], encoding="utf-8"))
assert complete["rows"] == 1100 and complete["unique_keys"] == 1100
assert audit["status"] == "PASS"
assert audit["detail_rows"] == 1100 and audit["broad_rows"] == 2200
assert audit["expected_broad_rows"] == 2200 and audit["hook_failures"] == 0
PY
  touch "$marker"
  phase "serial mediation complete"
}

run_induction() {
  local output="$RUN_ROOT/induction_v3"
  local log="$RUN_ROOT/logs/induction_v3.log"
  local marker="$output/Gemma4-E4B/.RUN_AND_ANALYSIS_COMPLETE"
  if [[ -f "$marker" ]]; then
    phase "induction follow-up already complete"
    return
  fi
  phase "induction follow-up start"
  mkdir -p "$output"
  "$PYTHON" scripts/run_realistic_niah_v4_4_5_induction_circuit.py \
    --model Gemma4-E4B \
    --stimuli "$STIMULI" \
    --v4-config configs/realistic_niah_v4_4_5_stimuli.json \
    --experiment-config "$INDUCTION_CONFIG" \
    --output-dir "$output" \
    --cache-dir "$CACHE_DIR" \
    --device-map cuda \
    >"$log" 2>&1
  "$PYTHON" scripts/analyze_realistic_niah_v4_4_5_induction_circuit.py \
    --run-root "$output" \
    --models Gemma4-E4B \
    --bootstrap-draws 10000 \
    --experiment-config "$INDUCTION_CONFIG" \
    >>"$log" 2>&1
  "$PYTHON" - "$output/Gemma4-E4B/complete.json" "$output/Gemma4-E4B/analysis_audit.json" <<'PY'
import json
import sys

complete = json.load(open(sys.argv[1], encoding="utf-8"))
audit = json.load(open(sys.argv[2], encoding="utf-8"))
assert complete["status"] == "complete"
assert complete["synthetic_rows"] == 1200 and complete["canonical_rows"] == 300
assert audit["status"] == "PASS" and audit["rows"] == 300
PY
  touch "$marker"
  phase "induction follow-up complete"
}

run_noise() {
  local output="$RUN_ROOT/noise_factorial_v2"
  local log="$RUN_ROOT/logs/noise_factorial_v2.log"
  local marker="$output/Gemma4-E4B/.RUN_AND_ANALYSIS_COMPLETE"
  if [[ -f "$marker" ]]; then
    phase "noise/outside-context follow-up already complete"
    return
  fi
  phase "noise/outside-context follow-up start"
  mkdir -p "$output"
  "$PYTHON" scripts/run_realistic_niah_v4_4_5_noise_factorial.py \
    --model Gemma4-E4B \
    --stimuli "$STIMULI" \
    --v4-config configs/realistic_niah_v4_4_5_stimuli.json \
    --experiment-config "$NOISE_CONFIG" \
    --output-dir "$output" \
    --cache-dir "$CACHE_DIR" \
    --device-map cuda \
    >"$log" 2>&1
  "$PYTHON" scripts/analyze_realistic_niah_v4_4_5_noise_factorial.py \
    --run-root "$output" \
    --models Gemma4-E4B \
    --bootstrap-draws 10000 \
    >>"$log" 2>&1
  "$PYTHON" - "$output/Gemma4-E4B/complete.json" "$output/Gemma4-E4B/analysis_audit.json" <<'PY'
import json
import sys

complete = json.load(open(sys.argv[1], encoding="utf-8"))
audit = json.load(open(sys.argv[2], encoding="utf-8"))
assert complete["status"] == "complete"
assert complete["factorial_rows"] == 240 and complete["outside_context_rows"] == 400
assert audit["status"] == "PASS"
PY
  touch "$marker"
  phase "noise/outside-context follow-up complete"
}

run_behavior_invariance_audit() {
  local output="$RUN_ROOT/behavior_invariance_audit.json"
  phase "Top-6 auxiliary-readout behavior-invariance audit start"
  test -d "$ORIGINAL_INDUCTION"
  test -d "$ORIGINAL_NOISE"
  "$PYTHON" scripts/audit_realistic_niah_v4_4_5_top6_behavior_invariance.py \
    --old-induction-dir "$ORIGINAL_INDUCTION" \
    --old-noise-dir "$ORIGINAL_NOISE" \
    --new-root "$RUN_ROOT" \
    --output "$output"
  "$PYTHON" - "$output" <<'PY'
import json
import sys

audit = json.load(open(sys.argv[1], encoding="utf-8"))
assert audit["status"] == "PASS"
assert len(audit["comparisons"]) == 4
assert all(row["non_bank_behavior_exactly_equal"] for row in audit["comparisons"])
PY
  phase "Top-6 auxiliary-readout behavior-invariance audit complete"
}

run_original_preservation_audit() {
  local output="$RUN_ROOT/original_preservation_audit.json"
  phase "historical-output preservation audit start"
  "$PYTHON" scripts/audit_realistic_niah_v4_4_5_original_preservation.py \
    --baseline "$PRESERVATION_SNAPSHOT" \
    --output "$output"
  "$PYTHON" - "$output" <<'PY'
import json
import sys

audit = json.load(open(sys.argv[1], encoding="utf-8"))
assert audit["status"] == "PASS"
assert len(audit["comparisons"]) == 14
assert all(row["unchanged"] for row in audit["comparisons"])
PY
  phase "historical-output preservation audit complete"
}

run_retrieval 29
run_serial
run_retrieval 35
run_induction
run_noise
run_behavior_invariance_audit
run_original_preservation_audit

sha256sum \
  "$SPAN_CONFIG" "$SERIAL_CONFIG" "$INDUCTION_CONFIG" "$NOISE_CONFIG" \
  scripts/analyze_realistic_niah_v4_4_5_retrieval_geometry.py \
  scripts/analyze_realistic_niah_v4_4_5_topk_attention_response.py \
  scripts/run_realistic_niah_v4_4_5_retrieval_subspace.py \
  scripts/run_realistic_niah_v4_4_5_serial_mediation.py \
  scripts/run_realistic_niah_v4_4_5_induction_circuit.py \
  scripts/run_realistic_niah_v4_4_5_noise_factorial.py \
  scripts/audit_realistic_niah_v4_4_5_top6_behavior_invariance.py \
  scripts/audit_realistic_niah_v4_4_5_original_preservation.py \
  scripts/supervise_realistic_niah_v4_4_5_gemma_top6_followups.sh \
  >"$RUN_ROOT/provenance/input_sha256.txt"
git rev-parse HEAD >"$RUN_ROOT/provenance/git_sha.txt"
"$PYTHON" - "$RUN_ROOT" <<'PY'
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
payload = {
    "status": "PASS",
    "schema_version": "realistic_niah_v4_4_5_gemma_top6_followups_v1",
    "model": "Gemma4-E4B",
    "global_frozen_top6": [
        [29, 4], [35, 2], [35, 7], [35, 1], [35, 3], [29, 2]
    ],
    "retrieval_layer_heads": {"29": [4, 2], "35": [2, 7, 1, 3]},
    "stimulus_sha256": "da4dd86142eb8a07f9a7e53497efd3375184c8e68367d4db994370fcb331f090",
    "completed": [
        "config_diff_audit",
        "attention_response",
        "retrieval_geometry",
        "retrieval_subspace_L29",
        "serial_mediation",
        "retrieval_subspace_L35",
        "induction_broad_readout",
        "noise_outside_context_broad_readout",
        "behavior_invariance_audit",
        "original_preservation_audit",
    ],
    "original_outputs_overwritten": False,
}
(root / "FINAL_AUDIT.json").write_text(
    json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
)
PY
touch "$RUN_ROOT/.ALL_COMPLETE"
phase "ALL GEMMA TOP-6 FOLLOW-UPS COMPLETE"
