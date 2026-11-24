@echo off
setlocal enabledelayedexpansion
title Stop Sequence Detection

cd /d "%~dp0"

where powershell >nul 2>&1
if %ERRORLEVEL% NEQ 0 (
    echo [ERROR] PowerShell is required but was not found.
    pause
    exit /b 1
)

powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0sequence_backend\stop_dev_windows.ps1" %*

echo.
pause
