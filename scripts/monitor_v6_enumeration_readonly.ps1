$ErrorActionPreference = 'Stop'

$sshExe = 'C:\Windows\System32\OpenSSH\ssh.exe'
$sshKeygenExe = 'C:\Windows\System32\OpenSSH\ssh-keygen.exe'
$identityFile = 'C:\Users\HP\.ssh\lambda_ed25519'
$knownHostsFile = 'C:\Users\HP\.ssh\known_hosts_filestream_68.209.73.209'
$expectedHostFingerprint = 'SHA256:YsSdiKivUKGMnyaACzHWKglFKVDiBQ1ZRVrVNARYrLw'
$remote = 'ubuntu@68.209.73.209'

$knownHostDescription = & $sshKeygenExe -lf $knownHostsFile
if ($LASTEXITCODE -ne 0 -or
    $knownHostDescription -notmatch [regex]::Escape($expectedHostFingerprint)) {
    throw "Pinned filestream host key does not match $expectedHostFingerprint"
}

# This monitor is intentionally parameter-free and read-only. Keep the remote
# command fixed so a reusable local permission cannot be repurposed to execute
# arbitrary SSH actions.
$remoteCommand = @'
date -Is
ssh-keygen -lf /etc/ssh/ssh_host_ed25519_key.pub
echo FINAL_MARKERS
test -e /home/ubuntu/CoT-Native-thinking-v5/runs/v6_enumeration_replication_20260828/queue_logs/Qwen3-8B_confirmation.COMPLETE && echo PRESENT:Qwen3-8B_confirmation.COMPLETE || echo MISSING:Qwen3-8B_confirmation.COMPLETE
test -e /home/ubuntu/CoT-Native-thinking-v5/runs/v6_enumeration_replication_20260828/queue_logs/Gemma4-E4B_confirmation.COMPLETE && echo PRESENT:Gemma4-E4B_confirmation.COMPLETE || echo MISSING:Gemma4-E4B_confirmation.COMPLETE
test -e /home/ubuntu/CoT-Native-thinking-v5/runs/v6_enumeration_replication_20260828/native_aligned_representation/COMPLETE && echo PRESENT:native_aligned_representation/COMPLETE || echo MISSING:native_aligned_representation/COMPLETE
test -e /home/ubuntu/CoT-Native-thinking-v5/runs/v6_enumeration_replication_20260828/final_audit/suite_completion.COMPLETE && echo PRESENT:final_audit/suite_completion.COMPLETE || echo MISSING:final_audit/suite_completion.COMPLETE
test -e /home/ubuntu/CoT-Native-thinking-v5/runs/v6_enumeration_replication_20260828/queue_logs/final_audit.COMPLETE && echo PRESENT:queue_logs/final_audit.COMPLETE || echo MISSING:queue_logs/final_audit.COMPLETE
test -e /home/ubuntu/CoT-Native-thinking-v5/runs/v6_enumeration_replication_20260828/final_report/NiaH_V6_Index_Bullet_Replication_report.COMPLETE && echo PRESENT:final_report/report.COMPLETE || echo MISSING:final_report/report.COMPLETE
test -e /home/ubuntu/CoT-Native-thinking-v5/runs/v6_enumeration_replication_20260828/queue_logs/final_report.COMPLETE && echo PRESENT:queue_logs/final_report.COMPLETE || echo MISSING:queue_logs/final_report.COMPLETE
echo EXTENSION_MARKERS
grep -qx PASS /home/ubuntu/CoT-Native-thinking-v5/runs/v6_enumeration_replication_20260828/queue_logs/index_item_end_anchor_sensitivity.COMPLETE 2>/dev/null && echo PASS:index_item_end_anchor_sensitivity.COMPLETE || echo PENDING:index_item_end_anchor_sensitivity.COMPLETE
grep -qx PASS /home/ubuntu/CoT-Native-thinking-v5/runs/v6_enumeration_replication_20260828/queue_logs/answer_trace_extension.COMPLETE 2>/dev/null && echo PASS:answer_trace_extension.COMPLETE || echo PENDING:answer_trace_extension.COMPLETE
grep -qx PASS /home/ubuntu/CoT-Native-thinking-v5/runs/v6_enumeration_replication_20260828/answer_trace_extension_report/report.COMPLETE 2>/dev/null && echo PASS:answer_trace_extension_report/report.COMPLETE || echo PENDING:answer_trace_extension_report/report.COMPLETE
grep -qx PASS /home/ubuntu/CoT-Native-thinking-v5/runs/v6_enumeration_replication_20260828/enumeration_index/Qwen3-8B/causal/answer_trace_extension_v1/extension.COMPLETE 2>/dev/null && echo PASS:index/Qwen3-8B/extension.COMPLETE || echo PENDING:index/Qwen3-8B/extension.COMPLETE
grep -qx PASS /home/ubuntu/CoT-Native-thinking-v5/runs/v6_enumeration_replication_20260828/enumeration_bullet/Qwen3-8B/causal/answer_trace_extension_v1/extension.COMPLETE 2>/dev/null && echo PASS:bullet/Qwen3-8B/extension.COMPLETE || echo PENDING:bullet/Qwen3-8B/extension.COMPLETE
grep -qx PASS /home/ubuntu/CoT-Native-thinking-v5/runs/v6_enumeration_replication_20260828/enumeration_index/Gemma4-E4B/causal/answer_trace_extension_v1/extension.COMPLETE 2>/dev/null && echo PASS:index/Gemma4-E4B/extension.COMPLETE || echo PENDING:index/Gemma4-E4B/extension.COMPLETE
grep -qx PASS /home/ubuntu/CoT-Native-thinking-v5/runs/v6_enumeration_replication_20260828/enumeration_bullet/Gemma4-E4B/causal/answer_trace_extension_v1/extension.COMPLETE 2>/dev/null && echo PASS:bullet/Gemma4-E4B/extension.COMPLETE || echo PENDING:bullet/Gemma4-E4B/extension.COMPLETE
echo SENSITIVITY_EVENTS
tail -n 180 /home/ubuntu/CoT-Native-thinking-v5/runs/v6_enumeration_replication_20260828/queue_logs/index_item_end_anchor_sensitivity.log 2>/dev/null | grep -E '^\[[^]]+\] (START|PASS|FAIL|COMPLETE|REUSE)|^\[v5 causal-(source-writes|heads-behavior)\]|Traceback|RuntimeError|CUDA out of memory|ValueError:' | tail -n 45 || true
echo ANSWER_TRACE_EVENTS
tail -n 180 /home/ubuntu/CoT-Native-thinking-v5/runs/v6_enumeration_replication_20260828/queue_logs/answer_trace_extension.log 2>/dev/null | grep -E '^\[[^]]+\] (START|PASS|FAIL|COMPLETE|WAIT|READY|REUSE)|Traceback|RuntimeError|CUDA out of memory|ValueError:' | tail -n 45 || true
echo ANSWER_TRACE_CELL_EVENTS
find /home/ubuntu/CoT-Native-thinking-v5/runs/v6_enumeration_replication_20260828 -path '*/causal/answer_trace_extension_v1/logs/supervisor.log' -type f -print -exec tail -n 25 {} \; 2>/dev/null || true
echo QWEN_EVENTS
tail -n 600 /home/ubuntu/CoT-Native-thinking-v5/runs/v6_enumeration_replication_20260828/queue_logs/Qwen3-8B_confirmation.log 2>/dev/null | grep -E '^\[[^]]+\] (START|PASS|FAIL|COMPLETE|WAIT|REUSE)|^\[(count-stream|replacement|v6 coherent)|Traceback|RuntimeError|CUDA out of memory|ValueError:|"native_loop_policy_freeze"' | tail -n 30 || true
echo QWEN_REPORT_TAIL_STAGE_TIMINGS
grep -E '^\[[^]]+\] (START|PASS) (native_loop_coherent_panel|native_loop_plan|native_loop_p0|native_loop_boundary|native_loop_analysis|restoration|restoration_analysis|single_seed_walkthrough|single_seed_walkthrough_analysis)' /home/ubuntu/CoT-Native-thinking-v5/runs/v6_enumeration_replication_20260828/queue_logs/Qwen3-8B_confirmation.log 2>/dev/null | tail -n 40 || true
echo GEMMA_EVENTS
tail -n 600 /home/ubuntu/CoT-Native-thinking-v5/runs/v6_enumeration_replication_20260828/queue_logs/Gemma4-E4B_confirmation.log 2>/dev/null | grep -E '^\[[^]]+\] (START|PASS|FAIL|COMPLETE|WAIT|REUSE)|^\[(count-stream|replacement|v6 coherent)|Traceback|RuntimeError|CUDA out of memory|ValueError:|"native_loop_policy_freeze"' | tail -n 50 || true
echo GEMMA_REPORT_TAIL_STAGE_TIMINGS
grep -E '^\[[^]]+\] (START|PASS) (native_loop_coherent_panel|native_loop_plan|native_loop_p0|native_loop_boundary|native_loop_analysis|restoration|restoration_analysis|single_seed_walkthrough|single_seed_walkthrough_analysis)' /home/ubuntu/CoT-Native-thinking-v5/runs/v6_enumeration_replication_20260828/queue_logs/Gemma4-E4B_confirmation.log 2>/dev/null | tail -n 40 || true
echo GEMMA_NATIVE_PANEL_RESOLUTION
grep -F '"accepted_replacement_seed_by_slot"' /home/ubuntu/CoT-Native-thinking-v5/runs/v6_enumeration_replication_20260828/queue_logs/Gemma4-E4B_confirmation.log 2>/dev/null | tail -n 1 || true
echo GEMMA_RECOVERY_INPUTS
find /home/ubuntu/CoT-Native-thinking-v5/runs/v6_enumeration_replication_20260828 -maxdepth 2 -type f -name stimuli.jsonl -print | sort
find /home/ubuntu/CoT-Native-thinking-v5/runs/v6_enumeration_replication_20260828 -maxdepth 1 -type d -name 'replacement_seed_pool*' -print | sort
sha256sum /home/ubuntu/CoT-Native-thinking-v5/code/Realistic_CoT_NiaH_Count_v6_20260828/configs/realistic_niah_v6_replacement_policy_amendment2.json
sha256sum /home/ubuntu/CoT-Native-thinking-v5/code/Realistic_CoT_NiaH_Count_v6_20260828/configs/realistic_niah_v6_coherent_native_loop_replacement_policy_amendment1.json
echo FINAL_AUDIT_EVENTS
tail -n 160 /home/ubuntu/CoT-Native-thinking-v5/runs/v6_enumeration_replication_20260828/queue_logs/final_audit.log 2>/dev/null || true
echo COHERENT_MANIFEST_POLICY_FIELDS
find /home/ubuntu/CoT-Native-thinking-v5/runs/v6_enumeration_replication_20260828 -path '*/replacement/*/manifest.json' -type f -print | sort | while read -r manifest; do
    printf '%s ' "$manifest"
    grep -oE '(coherent_policy|coherent_broad_policy|coherent_native_loop_policy)(_sha256)?[^,}]+' "$manifest" | tr '\n' ' '
    echo
done
echo FINAL_REPORT_EVENTS
tail -n 120 /home/ubuntu/CoT-Native-thinking-v5/runs/v6_enumeration_replication_20260828/queue_logs/final_report.log 2>/dev/null || true
echo ACTIVE_PROCESSES
pgrep -af 'supervise_realistic_niah_v6|run_realistic_niah_v6|queue_realistic_niah_v6' | tail -n 50 || true
echo TMUX
tmux list-sessions 2>/dev/null || true
echo GPU
nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu,temperature.gpu --format=csv,noheader,nounits
'@

$sshArgs = @(
    '-o', 'BatchMode=yes',
    '-o', 'IdentitiesOnly=yes',
    '-o', 'ConnectTimeout=15',
    '-o', 'StrictHostKeyChecking=yes',
    '-o', "UserKnownHostsFile=$knownHostsFile",
    '-i', $identityFile,
    $remote,
    $remoteCommand
)

& $sshExe @sshArgs
if ($LASTEXITCODE -ne 0) {
    throw "Read-only V6 monitor failed with SSH exit code $LASTEXITCODE"
}
