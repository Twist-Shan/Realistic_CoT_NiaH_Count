#!/usr/bin/env bash
set -euo pipefail

repo="${REPO:-/home/ubuntu/Realistic_CoT_NiaH_Count}"
root="${RUN_ROOT:-/home/ubuntu/runs/nonthinking_v445_8gpu_20260813}"
stimuli="$root/dataset/canonical_run_20260731_v4_numeric_presentation_v3_stimuli.jsonl"
basis="$root/analysis/retrieval_geometry/retrieval_bases.pt"
restoration="$root/canonical_merged"
run_root="$root/retrieval_subspace"
analysis_root="$root/analysis/retrieval_subspace"
cache="${HF_CACHE_DIR:-/home/ubuntu/hf-cache}"
logs="$root/logs/retrieval_subspace"
locks="$root/locks/retrieval_subspace"
seeds="1254,1255,1256,1257,1258,1259,1260,1261,1262,1263"
counts="1,2,3,4,5,6,7,8,9,10"
expected_stimuli_sha="da4dd86142eb8a07f9a7e53497efd3375184c8e68367d4db994370fcb331f090"

mkdir -p "$run_root" "$analysis_root" "$logs" "$locks"
exec 8>"$root/locks/retrieval_subspace_8gpu_supervisor.lock"
if ! flock -n 8; then
  echo "retrieval-subspace 8-GPU supervisor is already active" >&2
  exit 75
fi
cd "$repo"

test -f "$basis"
test -f "$restoration/Qwen3-8B/detail.jsonl"
test -f "$restoration/Gemma4-E4B/detail.jsonl"

actual_stimuli_sha="$(sha256sum "$stimuli" | cut -d' ' -f1)"
if [[ "$actual_stimuli_sha" != "$expected_stimuli_sha" ]]; then
  echo "Canonical stimulus hash audit failed: $actual_stimuli_sha != $expected_stimuli_sha" >&2
  exit 1
fi

PYTHONPATH=src .venv/bin/python - "$basis" <<'PY'
import sys
import torch

path = sys.argv[1]
bases = torch.load(path, map_location="cpu", weights_only=True)
expected = {
    "Qwen3-8B.L21", "Qwen3-8B.L23", "Qwen3-8B.L24",
    "Qwen3-8B.L26", "Qwen3-8B.L27",
    "Gemma4-E4B.L29", "Gemma4-E4B.L35",
}
missing = sorted(expected - set(bases))
if missing:
    raise SystemExit(f"retrieval basis missing keys: {missing}")
for key in sorted(expected):
    components = bases[key]["components"]
    if tuple(components.shape[:1]) != (3,):
        raise SystemExit(f"{key} is not rank 3: {tuple(components.shape)}")
print("retrieval basis preflight PASS", flush=True)
PY

run_one() {
  local gpu="$1"
  local model="$2"
  local retrieval_layer="$3"
  local label="$4"
  local output="$run_root/$label"
  local analysis="$analysis_root/$label"
  local log="$logs/${label}.log"
  local complete_marker="$output/.RUN_AND_ANALYSIS_COMPLETE"

  (
    exec 9>"$locks/${label}.lock"
    if ! flock -n 9; then
      echo "$label lock is already held" >&2
      exit 75
    fi
    if [[ -f "$complete_marker" ]]; then
      echo "$label already complete"
      exit 0
    fi

    export CUDA_VISIBLE_DEVICES="$gpu"
    export PYTHONPATH=src
    .venv/bin/python scripts/run_realistic_niah_v4_4_5_retrieval_subspace.py \
      --model "$model" \
      --stimuli "$stimuli" \
      --stimuli-config configs/realistic_niah_v4.json \
      --experiment-config configs/realistic_niah_v4_4_5_span_restoration_canonical.json \
      --basis-file "$basis" \
      --retrieval-layer "$retrieval_layer" \
      --source-patch-layer 8 \
      --seeds "$seeds" \
      --counts "$counts" \
      --output-dir "$output" \
      --cache-dir "$cache"

    .venv/bin/python - "$output/$model/detail.jsonl" "$model" "$retrieval_layer" <<'PY'
import json
import sys

path, expected_model, expected_layer = sys.argv[1], sys.argv[2], int(sys.argv[3])
rows = [json.loads(line) for line in open(path, encoding="utf-8") if line.strip()]
keys = {(int(r["seed"]), int(r["gold_count"]), str(r["condition"])) for r in rows}
expected = {
    (seed, count, condition)
    for seed in range(1254, 1264)
    for count in range(1, 11)
    for condition in (
        "clean_aligned_block", "clean_orthogonal_block",
        "restored_aligned_block", "restored_orthogonal_block",
    )
}
assert len(rows) == 400, (len(rows), path)
assert keys == expected, (len(keys), len(expected), sorted(expected - keys)[:5])
assert all(r["model_label"] == expected_model for r in rows)
assert all(int(r["retrieval_layer"]) == expected_layer for r in rows)
assert all(int(r["retrieval_block_applications"]) == 2 for r in rows)
assert all(
    int(r["source_patch_applications"]) == (2 if str(r["condition"]).startswith("restored") else 0)
    for r in rows
)
print(f"detail audit PASS rows={len(rows)} unique_keys={len(keys)}", flush=True)
PY

    .venv/bin/python scripts/analyze_realistic_niah_v4_4_5_retrieval_subspace.py \
      --run-root "$output" \
      --restoration-root "$restoration" \
      --output-dir "$analysis" \
      --models "$model"

    .venv/bin/python - "$analysis/analysis_audit.json" <<'PY'
import json
import sys

audit = json.load(open(sys.argv[1], encoding="utf-8"))
if audit.get("status") != "PASS":
    raise SystemExit(f"analysis audit failed: {audit}")
print("analysis audit PASS", flush=True)
PY
    touch "$complete_marker"
    echo "$label COMPLETE"
  ) >"$log" 2>&1
}

specs=(
  "0 Qwen3-8B 21 Qwen_L21"
  "1 Qwen3-8B 23 Qwen_L23"
  "2 Qwen3-8B 24 Qwen_L24"
  "3 Qwen3-8B 26 Qwen_L26"
  "4 Qwen3-8B 27 Qwen_L27"
  "5 Gemma4-E4B 29 Gemma_L29"
  "6 Gemma4-E4B 35 Gemma_L35"
)

pids=()
labels=()
for spec in "${specs[@]}"; do
  read -r gpu model layer label <<<"$spec"
  run_one "$gpu" "$model" "$layer" "$label" &
  pids+=("$!")
  labels+=("$label")
done

failed=0
for index in "${!pids[@]}"; do
  if ! wait "${pids[$index]}"; then
    echo "${labels[$index]} FAILED; inspect $logs/${labels[$index]}.log" >&2
    failed=1
  fi
done
if [[ "$failed" -ne 0 ]]; then
  exit 1
fi

.venv/bin/python - "$run_root" "$analysis_root" "$root/retrieval_subspace_8gpu_complete.json" "$actual_stimuli_sha" <<'PY'
import json
import sys
from pathlib import Path

run_root = Path(sys.argv[1])
analysis_root = Path(sys.argv[2])
output = Path(sys.argv[3])
stimulus_sha = sys.argv[4]
specs = (
    ("Qwen_L21", "Qwen3-8B", 21),
    ("Qwen_L23", "Qwen3-8B", 23),
    ("Qwen_L24", "Qwen3-8B", 24),
    ("Qwen_L26", "Qwen3-8B", 26),
    ("Qwen_L27", "Qwen3-8B", 27),
    ("Gemma_L29", "Gemma4-E4B", 29),
    ("Gemma_L35", "Gemma4-E4B", 35),
)
runs = []
for label, model, layer in specs:
    detail_path = run_root / label / model / "detail.jsonl"
    rows = [json.loads(line) for line in detail_path.open(encoding="utf-8") if line.strip()]
    audit_path = analysis_root / label / "analysis_audit.json"
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    if len(rows) != 400 or audit.get("status") != "PASS":
        raise SystemExit(
            f"combined audit failed for {label}: rows={len(rows)} status={audit.get('status')}"
        )
    runs.append(
        {
            "label": label,
            "model_label": model,
            "retrieval_layer": layer,
            "rows": len(rows),
            "analysis_status": audit["status"],
        }
    )

payload = {
    "status": "PASS",
    "schema_version": "realistic_niah_v4_4_5_retrieval_subspace_8gpu_v1",
    "stimulus_sha256": stimulus_sha,
    "source_patch_layer": 8,
    "confirmation_seeds": list(range(1254, 1264)),
    "counts": list(range(1, 11)),
    "conditions_per_seed_count": 4,
    "rows_per_model_layer": 400,
    "total_rows": sum(item["rows"] for item in runs),
    "runs": runs,
}
temporary = output.with_suffix(output.suffix + ".tmp")
temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
temporary.replace(output)
print(json.dumps(payload, indent=2, sort_keys=True))
PY
