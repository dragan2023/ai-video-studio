@echo off
setlocal
cd /d "%~dp0"
set "LOG=%~dp0Studio-startup.log"
echo [%date% %time%] Starting Studio > "%LOG%"
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0Start-Studio.ps1" >> "%LOG%" 2>&1
set "EXITCODE=%ERRORLEVEL%"
if not "%EXITCODE%"=="0" (
  echo.>> "%LOG%"
  echo Studio startup failed with exit code %EXITCODE%.>> "%LOG%"
  start "" notepad.exe "%LOG%"
  pause
  exit /b %EXITCODE%
)
