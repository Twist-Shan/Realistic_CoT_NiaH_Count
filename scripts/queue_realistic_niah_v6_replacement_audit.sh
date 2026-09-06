#!/usr/bin/env bash
set -euo pipefail

ROOT=${V6_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}
PYTHON=${V6_PYTHON:-$ROOT/.venv/bin/python}
RUN_BASE=${V6_RUN_BASE:-$ROOT/work/realistic_niah_v6}
POOL=${V6_REPLACEMENT_POOL:-$RUN_BASE/replacement_seed_pool}
OUTPUT=$RUN_BASE/replacement_audit
mkdir -p "$OUTPUT"
exec > >(tee -a "$OUTPUT/queue.log") 2>&1

for mode in enumeration_index enumeration_bullet; do
  for model in Qwen3-8B Gemma4-E4B; do
    marker=$RUN_BASE/$mode/$model/replacement/discovery/discovery.COMPLETE
    echo "[$(date --iso-8601=seconds)] WAIT $mode/$model replacement marker=$marker"
    while [[ ! -s "$marker" ]]; do sleep 30; done
    grep -qx PASS "$marker" || { echo "$marker is not PASS" >&2; exit 1; }
  done
done

"$PYTHON" "$ROOT/scripts/build_realistic_niah_v6_replacement_audit.py" \
  --run-base "$RUN_BASE" \
  --replacement-policy "$ROOT/configs/realistic_niah_v6_replacement_policy.json" \
  --pool-manifest "$POOL/manifest.json" --output "$OUTPUT"
printf 'PASS\n' >"$OUTPUT/audit.COMPLETE"
