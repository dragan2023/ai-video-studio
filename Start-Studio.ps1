# Starts Nautilus Studio and imports the adjacent .env file.
$ErrorActionPreference = "Stop"
$ProjectRoot = $PSScriptRoot
$StudioUrl = "http://127.0.0.1:7860"

function Fail-Start([string]$Message) {
    Write-Host ""
    Write-Host "Startup failed: $Message" -ForegroundColor Red
    exit 1
}

Set-Location -LiteralPath $ProjectRoot
$python = Join-Path $ProjectRoot ".venv\Scripts\python.exe"
if (-not (Test-Path -LiteralPath $python)) { Fail-Start "Missing .venv\Scripts\python.exe" }
if (-not (Test-Path -LiteralPath (Join-Path $ProjectRoot ".env"))) { Fail-Start "Missing .env" }
if (-not (Test-Path -LiteralPath (Join-Path $ProjectRoot "web\dist\index.html"))) { Fail-Start "Missing web\dist\index.html" }

# .env contains Chinese Windows paths; force UTF-8 instead of the legacy ANSI code page.
Get-Content -LiteralPath (Join-Path $ProjectRoot ".env") -Encoding utf8 | ForEach-Object {
    $line = $_.Trim()
    if ($line -match '^export\s+([A-Za-z_][A-Za-z0-9_]*)=(.*)$') {
        $name = $Matches[1]
        $value = $Matches[2].Trim()
        if ($value.Length -ge 2 -and (($value.StartsWith('"') -and $value.EndsWith('"')) -or ($value.StartsWith("'") -and $value.EndsWith("'")))) {
            $value = $value.Substring(1, $value.Length - 2)
        }
        [Environment]::SetEnvironmentVariable($name, $value, "Process")
    }
}
$env:PYTHONPATH = Join-Path $ProjectRoot "src"
$env:PYTHONUTF8 = "1"

# A previous one-click launch can outlive its console. Always stop leftover
# Studio processes on every run before starting a fresh instance:
#   1) any process running long_video_studio (uvicorn/python), and
#   2) whatever currently listens on Port 7860 if it is a python/studio process.
$studioProcesses = Get-CimInstance Win32_Process | Where-Object {
    $cl = [string]$_.CommandLine
    $cl -match "long_video_studio" -and ($cl -match "uvicorn" -or $cl -match "python")
}
$listenerPid = $null
$listener = Get-NetTCPConnection -LocalPort 7860 -State Listen -ErrorAction SilentlyContinue | Select-Object -First 1
if ($listener) { $listenerPid = [int]$listener.OwningProcess }

$toStop = @()
foreach ($proc in $studioProcesses) {
    if ($proc.ProcessId -notin $toStop) { $toStop += $proc.ProcessId }
}
if ($listenerPid -and $listenerPid -notin $toStop) {
    $procInfo = Get-CimInstance Win32_Process -Filter "ProcessId = $listenerPid" -ErrorAction SilentlyContinue
    $cl = [string]$procInfo.CommandLine
    if ($cl -match "python|uvicorn|long_video_studio") { $toStop += $listenerPid }
    else { Fail-Start "Port 7860 is occupied by an unrelated process (PID $listenerPid). It was not stopped." }
}

foreach ($procId in $toStop) {
    Write-Host "Stopping previous Studio process (PID $procId)..." -ForegroundColor Yellow
    Stop-Process -Id $procId -Force -ErrorAction SilentlyContinue
}
for ($attempt = 1; $attempt -le 10; $attempt++) {
    Start-Sleep -Milliseconds 500
    $stillListening = Get-NetTCPConnection -LocalPort 7860 -State Listen -ErrorAction SilentlyContinue | Select-Object -First 1
    if (-not $stillListening) { break }
}
if (Get-NetTCPConnection -LocalPort 7860 -State Listen -ErrorAction SilentlyContinue | Select-Object -First 1) {
    Fail-Start "Previous Studio process did not release port 7860."
}

Write-Host "Starting Nautilus Studio..." -ForegroundColor Cyan
Write-Host "Keep this window open while Studio is running." -ForegroundColor Yellow
Start-Job -ArgumentList $StudioUrl -ScriptBlock {
    param($Url)
    for ($attempt = 1; $attempt -le 45; $attempt++) {
        try {
            $response = Invoke-WebRequest -Uri "$Url/api/health" -UseBasicParsing -TimeoutSec 2
            if ($response.StatusCode -eq 200) { Start-Process $Url; return }
        } catch { }
        Start-Sleep -Seconds 1
    }
} | Out-Null

& $python -m uvicorn long_video_studio.app:create_app --factory --host 127.0.0.1 --port 7860
