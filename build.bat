@echo off
echo =======================================================
echo     Sequence Detection Build (Windows Desktop App)
echo =======================================================
cd /d "%~dp0sequence_backend"
powershell -ExecutionPolicy Bypass -File build_windows_desktop.ps1
if %ERRORLEVEL% NEQ 0 (
    echo.
    echo [ERROR] Build failed! Please check the logs above.
    pause
    exit /b %ERRORLEVEL%
)
echo.
echo =======================================================
echo Build completed successfully!
echo You can find the installer (.exe) in: sequence_frontend\dist_app
echo =======================================================
pause
