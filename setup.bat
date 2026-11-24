@echo off
setlocal enabledelayedexpansion
title Sequence Detection Setup

echo =======================================================
echo        Sequence Detection Setup (Windows)
echo =======================================================
echo.

cd /d "%~dp0"

where powershell >nul 2>&1
if %ERRORLEVEL% NEQ 0 (
    echo [ERROR] PowerShell is required to run the automated setup but was not found.
    pause
    exit /b 1
)

powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0sequence_backend\setup_windows.ps1" %*
if %ERRORLEVEL% NEQ 0 (
    echo.
    echo =======================================================
    echo [ERROR] Setup encountered an issue. Review the logs above.
    echo =======================================================
    pause
    exit /b %ERRORLEVEL%
)

echo.
echo =======================================================
echo [SUCCESS] Setup completed successfully!
echo To launch Sequence Detection, simply double-click run.bat
echo =======================================================
pause
