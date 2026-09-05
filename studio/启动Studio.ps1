# 启动 Nautilus Studio，并自动读取同目录 .env 配置。
$ErrorActionPreference = "Stop"
$ProjectRoot = $PSScriptRoot
$StudioUrl = "http://127.0.0.1:7860"

function Fail-Start([string]$Message) {
    Write-Host "" 
    Write-Host "启动失败：$Message" -ForegroundColor Red
    Write-Host "请按任意键关闭窗口。" -ForegroundColor Yellow
    [void][Console]::ReadKey($true)
    exit 1
}

Set-Location -LiteralPath $ProjectRoot

$python = Join-Path $ProjectRoot ".venv\Scripts\python.exe"
if (-not (Test-Path -LiteralPath $python)) {
    Fail-Start "未找到 .venv\Scripts\python.exe。请先完成 Studio 依赖安装。"
}
if (-not (Test-Path -LiteralPath (Join-Path $ProjectRoot ".env"))) {
    Fail-Start "未找到 .env 配置文件。"
}
if (-not (Test-Path -LiteralPath (Join-Path $ProjectRoot "web\dist\index.html"))) {
    Fail-Start "未找到 web\dist\index.html。请先构建前端。"
}

# .env 使用 bash 风格的 export KEY=VALUE；这里仅导入当前 Studio 进程所需变量。
Get-Content -LiteralPath (Join-Path $ProjectRoot ".env") | ForEach-Object {
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

# 避免重复启动；已有 Studio 时直接打开页面。
$listener = Get-NetTCPConnection -LocalPort 7860 -State Listen -ErrorAction SilentlyContinue | Select-Object -First 1
if ($listener) {
    Write-Host "检测到 7860 端口已在监听，直接打开 Studio：$StudioUrl" -ForegroundColor Yellow
    Start-Process $StudioUrl
    exit 0
}

Write-Host "正在启动 Nautilus Studio..." -ForegroundColor Cyan
Write-Host "浏览器会在服务就绪后自动打开。关闭本窗口会停止 Studio。" -ForegroundColor Yellow

Start-Job -ArgumentList $StudioUrl -ScriptBlock {
    param($Url)
    for ($attempt = 1; $attempt -le 45; $attempt++) {
        try {
            $response = Invoke-WebRequest -Uri "$Url/api/health" -UseBasicParsing -TimeoutSec 2
            if ($response.StatusCode -eq 200) {
                Start-Process $Url
                return
            }
        } catch { }
        Start-Sleep -Seconds 1
    }
} | Out-Null

& $python -m uvicorn long_video_studio.app:create_app --factory --host 127.0.0.1 --port 7860
