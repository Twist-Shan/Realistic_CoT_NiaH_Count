#!/usr/bin/env bash
set -euo pipefail

CODE_ROOT="${CODE_ROOT:-/home/ubuntu/CoT-Native-thinking-v5/code/Realistic_CoT_NiaH_Count_native_v5_p0_20260820}"
RUN_ROOT="${RUN_ROOT:-/home/ubuntu/CoT-Native-thinking-v5/runs/v5_native_unified_p0_20260820/Qwen3-8B}"
PYTHON="${PYTHON:-/home/ubuntu/CoT-Native-thinking-v5/venv/bin/python}"
CACHE_DIR="${CACHE_DIR:-/home/ubuntu/CoT-Native-thinking-v5/cache/huggingface}"
GENERATIONS="$CODE_ROOT/work/v5_trace_parser_v2/Qwen3-8B_generations_reparsed.jsonl"
CONFIG="$CODE_ROOT/configs/realistic_niah_v5.json"
OUTPUT="$RUN_ROOT/source_attention_p0_all_local_grammars_full_discovery_v2"
LOG="$RUN_ROOT/logs/qwen_unified_p0_full_discovery_source_supervisor.log"
COMPLETE="$RUN_ROOT/qwen_unified_p0_full_discovery_source_complete.json"

mkdir -p "$RUN_ROOT/logs" "$RUN_ROOT/locks" "$CACHE_DIR"
exec 9>"$RUN_ROOT/locks/qwen_unified_p0_full_discovery_source.lock"
if ! flock -n 9; then
  echo "another unified-P0 full-discovery source supervisor owns the lock" >&2
  exit 75
fi

cd "$CODE_ROOT"
echo "START source_writes utc=$(date -u +%FT%TZ)" | tee -a "$LOG"
env HF_HOME="$CACHE_DIR" TOKENIZERS_PARALLELISM=false \
  "$PYTHON" scripts/run_realistic_niah_v5.py causal-source-writes \
    --config "$CONFIG" \
    --model Qwen3-8B \
    --cache-dir "$CACHE_DIR" \
    --device-map auto \
    --torch-dtype bfloat16 \
    --attention-backend sdpa \
    --generations "$GENERATIONS" \
    --output "$OUTPUT" \
    --anchor-role p0_item_end \
    --include-secondary 2>&1 | tee -a "$LOG"
echo "COMPLETE source_writes utc=$(date -u +%FT%TZ)" | tee -a "$LOG"

"$PYTHON" - "$OUTPUT" "$COMPLETE" <<'PY'
import collections
import datetime
import json
import pathlib
import sys

output = pathlib.Path(sys.argv[1])
complete = pathlib.Path(sys.argv[2])
shards = sorted((output / "shards").glob("trial_*.jsonl"))
seeds = set()
grammars = collections.Counter()
local_rows = 0
for shard in shards:
    first = None
    with shard.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            if first is None:
                first = row
            if bool(row.get("local_anchor_eligible")):
                local_rows += 1
    if first is not None:
        seeds.add(int(first["seed"]))
        grammars[str(first.get("target_grammar_class"))] += 1

expected_seeds = set(range(1234, 1254))
assert seeds == expected_seeds, {
    "observed": sorted(seeds),
    "missing": sorted(expected_seeds - seeds),
}
manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
assert int(manifest["completed_shards"]) == len(shards)
payload = {
    "schema_version": "realistic_niah_v5_qwen_p0_full_discovery_source_v1",
    "status": "PASS",
    "completed_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
    "output": str(output),
    "completed_shards": len(shards),
    "eligible_anchor_tasks": int(manifest["eligible_anchor_tasks"]),
    "discovery_seed_count": len(seeds),
    "discovery_seeds": sorted(seeds),
    "target_grammar_anchor_counts": dict(sorted(grammars.items())),
    "local_eligible_head_rows": local_rows,
}
complete.write_text(
    json.dumps(payload, indent=2, sort_keys=True) + "\n",
    encoding="utf-8",
)
print(json.dumps(payload, sort_keys=True))
PY

echo "ALL_COMPLETE utc=$(date -u +%FT%TZ)" | tee -a "$LOG"
