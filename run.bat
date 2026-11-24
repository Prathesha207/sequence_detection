@echo off
setlocal enabledelayedexpansion
title Sequence Detection

cd /d "%~dp0"

where powershell >nul 2>&1
if %ERRORLEVEL% NEQ 0 (
    echo [ERROR] PowerShell is required to launch Sequence Detection but was not found.
    pause
    exit /b 1
)

powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0sequence_backend\run_dev_windows.ps1" %*
if %ERRORLEVEL% NEQ 0 (
    echo.
    echo =======================================================
    echo [ERROR] Failed to start Sequence Detection. Review logs above.
    echo =======================================================
    pause
    exit /b %ERRORLEVEL%
)

echo.
pause
