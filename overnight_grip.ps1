# Start the overnight grip job (open-grip search + grip lab + report), hidden, at below-normal priority.
#   powershell -ExecutionPolicy Bypass -File overnight_grip.ps1
# Stop it: Get-Process python | Where-Object { $_.Path -like "*Documents\Chopstix\.venv*" } | Stop-Process
$repo = $PSScriptRoot
$py = Join-Path $repo ".venv\Scripts\python.exe"
New-Item -ItemType Directory -Force (Join-Path $repo "runs\overnight") | Out-Null
$p = Start-Process -FilePath $py -ArgumentList "-m", "chopstix.tools.overnight_grip" -WorkingDirectory $repo -WindowStyle Hidden -PassThru `
    -RedirectStandardOutput (Join-Path $repo "runs\overnight\grip_stdout.txt") -RedirectStandardError (Join-Path $repo "runs\overnight\grip_stderr.txt")
$p.PriorityClass = "BelowNormal"
Write-Output "overnight grip job started, pid $($p.Id). Log: runs\overnight\grip_log.txt"
