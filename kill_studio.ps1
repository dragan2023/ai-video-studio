Get-CimInstance Win32_Process | Where-Object {
    $cl = [string]$_.CommandLine
    $cl -match 'long_video_studio' -and ($cl -match 'uvicorn' -or $cl -match 'python')
} | ForEach-Object {
    Write-Output ('killed ' + $_.ProcessId)
    Stop-Process -Id $_.ProcessId -Force -ErrorAction SilentlyContinue
}
