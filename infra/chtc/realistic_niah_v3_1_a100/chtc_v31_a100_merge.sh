#!/usr/bin/env bash
set -euo pipefail

expected_commit="0afd1adcf2e4b9033805323b9e396220c29afb9e"
work_root="${_CONDOR_SCRATCH_DIR:-$PWD}"
repo="${work_root}/repo"
run_id="chtc_a100_merged_${CLUSTER:-manual}_${PROCESS:-0}"
run_root="${repo}/runs/realistic_niah_v3_1/${run_id}"
result_archive="${PWD}/chtc_v31_a100_merged_result.tar.gz"
groups=(a100_80_0 a100_80_1 a100_40_0 a100_40_1 a100_40_2 a100_40_3)

archive_results() {
  status=$?
  set +e
  mkdir -p "${run_root}/orchestration"
  printf '{"exit_status":%s,"cluster":"%s","process":"%s","completed_at_utc":"%s"}\n' \
    "${status}" "${CLUSTER:-unknown}" "${PROCESS:-unknown}" \
    "$(date -u +%Y-%m-%dT%H:%M:%SZ)" \
    > "${run_root}/orchestration/chtc_a100_merge_exit.json"
  tar --exclude="runs/realistic_niah_v3_1/${run_id}/dataset" \
    -czf "${result_archive}" -C "${repo}" \
    "runs/realistic_niah_v3_1/${run_id}" 2>/dev/null || true
  exit "${status}"
}
trap archive_results EXIT

git clone -b codex/realistic-consolidation realistic_v31.bundle "${repo}"
[[ "$(git -C "${repo}" rev-parse HEAD)" == "${expected_commit}" ]]
mkdir -p "${run_root}"
tar -xzf realistic_v31_dataset.tar.gz -C "${run_root}"
export PYTHONPATH="${repo}/src"
python3 "${repo}/scripts/prepare_realistic_niah_v3_1.py" \
  --run-root "${run_root}" --repo-root "${repo}" \
  > "${run_root}/orchestration/merge_prepare_stdout.json"

for group in "${groups[@]}"; do
  archive="${work_root}/chtc_v31_part_${group}.tar.gz"
  test -s "${archive}"
  tar -tzf "${archive}" > "${run_root}/orchestration/${group}_archive_members.txt"
  tar -xzf "${archive}" -C "${run_root}"
  python3 -c \
    'import json,sys; p=json.load(open(sys.argv[1])); assert p["group_id"]==sys.argv[2] and p["exit_status"]==0' \
    "${run_root}/partial_meta/${group}/exit.json" "${group}"
  test -s "${run_root}/partial_meta/${group}/completed.json"
done

manifest_count="$(find "${run_root}/shards" -type f -name run_manifest.json | wc -l)"
[[ "${manifest_count}" -eq 48 ]]
python3 "${repo}/scripts/merge_realistic_niah_v3_1_shards.py" --run-root "${run_root}"
python3 -c \
  'import json,sys; a=json.load(open(sys.argv[1])); assert a["passed"] is True; assert a["requests"]==a["unique_request_ids"]==161280' \
  "${run_root}/orchestration/final_shard_audit.json"
python3 -m pip freeze > "${run_root}/orchestration/merge_pip_freeze.txt"
echo "merge_completed_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
