# Start Nautilus main workflow (7860) and Infinite Canvas workbench (3000).
# Expected workspace layout:
#   <workspace>/nautilus-studio/Start-DualFrontend.ps1
#   <workspace>/infinite-canvas/web/package.json
$ErrorActionPreference = "Stop"
$NautilusRoot = $PSScriptRoot
$WorkspaceRoot = Split-Path -Parent $NautilusRoot
$CanvasWeb = Join-Path $WorkspaceRoot "infinite-canvas\web"

function Fail-Start([string]$Message) {
    Write-Host "";
    Write-Host "Startup failed: $Message" -ForegroundColor Red
    exit 1
}

function Test-Listen([int]$Port) {
    return [bool](Get-NetTCPConnection -LocalPort $Port -State Listen -ErrorAction SilentlyContinue | Select-Object -First 1)
}

if (-not (Test-Path -LiteralPath (Join-Path $NautilusRoot "Start-Studio.ps1"))) {
    Fail-Start "Missing Nautilus Start-Studio.ps1"
}
if (-not (Test-Path -LiteralPath (Join-Path $CanvasWeb "package.json"))) {
    Fail-Start "Missing Infinite Canvas at $CanvasWeb"
}
if (-not (Get-Command npm -ErrorAction SilentlyContinue)) {
    Fail-Start "npm is not available in PATH"
}

# Nautilus owns batch production and must always be the main entry.
if (-not (Test-Listen 7860)) {
    Write-Host "Starting Nautilus main workflow on :7860 ..." -ForegroundColor Cyan
    Start-Process -FilePath "powershell.exe" -ArgumentList @(
        "-NoExit", "-ExecutionPolicy", "Bypass", "-File", (Join-Path $NautilusRoot "Start-Studio.ps1")
    ) | Out-Null
} else {
    Write-Host "Nautilus already listens on :7860" -ForegroundColor Green
}

# Canvas is a secondary workbench for assets and one-shot adjustments.
if (-not (Test-Listen 3000)) {
    Write-Host "Starting Infinite Canvas workbench on :3000 ..." -ForegroundColor Cyan
    $canvasCommand = "Set-Location -LiteralPath '$CanvasWeb'; npm run dev"
    Start-Process -FilePath "powershell.exe" -ArgumentList @("-NoExit", "-Command", $canvasCommand) | Out-Null
} else {
    Write-Host "Infinite Canvas already listens on :3000" -ForegroundColor Green
}

# Wait briefly, then open the main workflow first. Canvas is opened separately for asset/single-shot work.
for ($i = 1; $i -le 30; $i++) {
    if ((Test-Listen 7860) -and (Test-Listen 3000)) { break }
    Start-Sleep -Seconds 1
}

if (Test-Listen 7860) { Start-Process "http://127.0.0.1:7860" }
if (Test-Listen 3000) { Start-Process "http://localhost:3000/canvas" }

Write-Host "";
Write-Host "Main workflow: http://127.0.0.1:7860" -ForegroundColor Green
Write-Host "Canvas workbench: http://localhost:3000/canvas" -ForegroundColor Green
