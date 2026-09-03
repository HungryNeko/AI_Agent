$ErrorActionPreference = "Stop"

$root = Split-Path -Parent $MyInvocation.MyCommand.Path
$backendCommand = "Set-Location -LiteralPath '$root'; `$env:AI_AGENT_BACKEND_PORT='8012'; conda run --no-capture-output -n sde python backend\scripts\server.py"
$frontendCommand = "Set-Location -LiteralPath '$root\frontend'; npm.cmd run dev"

function Stop-DevPort {
    param([int]$Port)

    $seen = @{}
    $connections = Get-NetTCPConnection -LocalPort $Port -State Listen -ErrorAction SilentlyContinue
    foreach ($connection in $connections) {
        $processId = $connection.OwningProcess
        if ($processId) {
            $seen[$processId] = $true
        }
    }

    $netstatLines = netstat -ano | Select-String -Pattern "LISTENING"
    foreach ($line in $netstatLines) {
        $parts = ($line.ToString().Trim() -split "\s+")
        if ($parts.Count -lt 5) {
            continue
        }
        $localAddress = $parts[1]
        $processId = 0
        if (
            ($localAddress -eq "127.0.0.1:$Port" -or $localAddress -eq "0.0.0.0:$Port" -or $localAddress -eq "[::1]:$Port" -or $localAddress -eq "[::]:$Port") -and
            [int]::TryParse($parts[-1], [ref]$processId)
        ) {
            $seen[$processId] = $true
        }
    }

    foreach ($processId in $seen.Keys) {
        if ($processId -eq $PID -or $processId -eq 0) {
            continue
        }
        $process = Get-Process -Id $processId -ErrorAction SilentlyContinue
        if ($process) {
            Write-Host "Stopping old process on port ${Port}: $($process.ProcessName) ($processId)"
            Stop-Process -Id $processId -Force
        }
    }
}

Stop-DevPort 8010
Stop-DevPort 8011
Stop-DevPort 8012
Stop-DevPort 5173
Stop-DevPort 5174

Write-Host "Starting backend:  http://127.0.0.1:8012"
Start-Process powershell.exe -ArgumentList @(
    "-NoProfile",
    "-NoExit",
    "-ExecutionPolicy", "Bypass",
    "-Command", $backendCommand
) -WindowStyle Normal

Write-Host "Starting frontend: http://127.0.0.1:5173/"
Start-Process powershell.exe -ArgumentList @(
    "-NoProfile",
    "-NoExit",
    "-ExecutionPolicy", "Bypass",
    "-Command", $frontendCommand
) -WindowStyle Normal

Write-Host ""
Write-Host "Open: http://127.0.0.1:5173/"
Write-Host "Close the two PowerShell windows to stop the dev servers."
