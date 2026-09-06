param(
    [Parameter(Mandatory=$true)][string]$RemoteCommand,
    [string]$TargetHost = '68.209.74.143',
    [int]$TimeoutSeconds = 45
)
$ErrorActionPreference = 'Stop'
$repoRoot = Split-Path (Split-Path $PSScriptRoot -Parent) -Parent
$knownHosts = 'additional_experiments/runs/connectivity/known_hosts'
$startInfo = [System.Diagnostics.ProcessStartInfo]::new()
$startInfo.FileName = 'C:\Windows\System32\OpenSSH\ssh.exe'
$startInfo.WorkingDirectory = $repoRoot
$startInfo.UseShellExecute = $false
$startInfo.CreateNoWindow = $true
$startInfo.RedirectStandardOutput = $true
$startInfo.RedirectStandardError = $true
foreach ($argument in @('-n','-F','NUL','-i','C:\Users\HP\.ssh\lambda_ed25519',
    '-o','IdentitiesOnly=yes','-o','BatchMode=yes','-o','ConnectTimeout=15',
    '-o','ServerAliveInterval=10','-o','ServerAliveCountMax=2',
    '-o',"UserKnownHostsFile=$knownHosts", "ubuntu@$TargetHost", $RemoteCommand)) {
    $startInfo.ArgumentList.Add($argument)
}
$probe = [System.Diagnostics.Process]::new()
$probe.StartInfo = $startInfo
$null = $probe.Start()
$stdout = $probe.StandardOutput.ReadToEndAsync()
$stderr = $probe.StandardError.ReadToEndAsync()
if (-not $probe.WaitForExit($TimeoutSeconds * 1000)) {
    $probe.Kill()
    $probe.WaitForExit()
    Write-Output $stdout.GetAwaiter().GetResult()
    Write-Output $stderr.GetAwaiter().GetResult()
    throw "SSH command exceeded $TimeoutSeconds seconds; only its local client was terminated."
}
Write-Output $stdout.GetAwaiter().GetResult()
Write-Output $stderr.GetAwaiter().GetResult()
exit $probe.ExitCode
