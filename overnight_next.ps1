# Start the follow-up job (grip gate -> keep altering the open grip until >= 90% -> stage 1 and 2 training), hidden.
#   powershell -ExecutionPolicy Bypass -File overnight_next.ps1 [-WaitPid <pid of the grip job>]
# Stop it (and any training it started): Get-Process python | Where-Object { $_.Path -like "*Documents\Chopstix\.venv*" } | Stop-Process
param([int]$WaitPid = 0, [string]$Pair = "")
$repo = $PSScriptRoot
$py = Join-Path $repo ".venv\Scripts\python.exe"
New-Item -ItemType Directory -Force (Join-Path $repo "runs\overnight") | Out-Null
$p = Start-Process -FilePath $py -ArgumentList "-m", "chopstix.tools.overnight_next", "--wait-pid", $WaitPid, "--pair", "`"$Pair`"" -WorkingDirectory $repo -WindowStyle Hidden -PassThru `
    -RedirectStandardOutput (Join-Path $repo "runs\overnight\next_stdout.txt") -RedirectStandardError (Join-Path $repo "runs\overnight\next_stderr.txt")
Write-Output "follow-up job started, pid $($p.Id). Log: runs\overnight\next_log.txt"
