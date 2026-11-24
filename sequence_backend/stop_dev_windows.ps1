# ==============================================================================
# Sequence Detection - Windows Development Stopper
# Terminates any running backend (port 8000 / uvicorn / python) and frontend (port 5173 / vite / node)
# ==============================================================================
$BackendDir = if ($MyInvocation.MyCommand.Path) {
    Split-Path -Parent $MyInvocation.MyCommand.Path
} else {
    Get-Location
}
$TargetPort = if ($env:BACKEND_PORT) { [int]$env:BACKEND_PORT } elseif ($env:PORT) { [int]$env:PORT } else { 8000 }
$FrontendPort = 5173

Write-Host "=======================================================" -ForegroundColor Yellow
Write-Host "      Stopping Sequence Detection Servers...           " -ForegroundColor Yellow
Write-Host "=======================================================" -ForegroundColor Yellow

$stopped = 0

# 1. Stop process listening on backend port
try {
    $pids = Get-NetTCPConnection -LocalPort $TargetPort -State Listen -ErrorAction SilentlyContinue | Select-Object -ExpandProperty OwningProcess -Unique
    foreach ($p in $pids) {
        if ($p -and $p -gt 0 -and $p -ne $PID) {
            Write-Host "Terminating process on port $TargetPort (PID $p)..." -ForegroundColor Cyan
            Stop-Process -Id $p -Force -ErrorAction SilentlyContinue
            $stopped++
        }
    }
} catch {}

# 2. Stop process listening on frontend port
try {
    $pids5173 = Get-NetTCPConnection -LocalPort $FrontendPort -State Listen -ErrorAction SilentlyContinue | Select-Object -ExpandProperty OwningProcess -Unique
    foreach ($p in $pids5173) {
        if ($p -and $p -gt 0 -and $p -ne $PID) {
            Write-Host "Terminating process on port $FrontendPort (PID $p)..." -ForegroundColor Cyan
            Stop-Process -Id $p -Force -ErrorAction SilentlyContinue
            $stopped++
        }
    }
} catch {}

# 3. Stop Python backend processes matching .venv or app.main:app
try {
    $CimProcs = Get-CimInstance Win32_Process -Filter "Name like 'python%.exe' or Name='uvicorn.exe'" -ErrorAction SilentlyContinue
    foreach ($lp in $CimProcs) {
        if ($lp.ProcessId -ne $PID -and (($lp.ExecutablePath -like "*$BackendDir*") -or ($lp.CommandLine -like "*$BackendDir*") -or ($lp.CommandLine -like "*app.main:app*"))) {
            Write-Host "Terminating backend worker (PID $($lp.ProcessId))..." -ForegroundColor Cyan
            Stop-Process -Id $lp.ProcessId -Force -ErrorAction SilentlyContinue
            $stopped++
        }
    }
} catch {}

# 4. Close console windows
try {
    $CmdProcs = Get-CimInstance Win32_Process -Filter "Name='cmd.exe'" -ErrorAction SilentlyContinue
    foreach ($cp in $CmdProcs) {
        if ($cp.ProcessId -ne $PID -and (($cp.CommandLine -like "*Sequence Detection - Backend API*") -or ($cp.CommandLine -like "*Sequence Detection - Frontend UI*"))) {
            Write-Host "Closing console window (PID $($cp.ProcessId))..." -ForegroundColor Cyan
            Stop-Process -Id $cp.ProcessId -Force -ErrorAction SilentlyContinue
            $stopped++
        }
    }
} catch {}

Write-Host ""
if ($stopped -gt 0) {
    Write-Host "[SUCCESS] Stopped $stopped server process(es)." -ForegroundColor Green
} else {
    Write-Host "[INFO] No running Sequence Detection servers found." -ForegroundColor Green
}
Write-Host "=======================================================" -ForegroundColor Yellow
