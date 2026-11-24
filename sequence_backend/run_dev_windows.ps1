# ==============================================================================
# Sequence Detection - Windows Development Runner
# Auto-cleans lingering servers, verifies venv, packages, CUDA kernels,
# and starts backend (FastAPI/Uvicorn) and frontend (Vite) cleanly.
# ==============================================================================
$ErrorActionPreference = 'Continue'

$BackendDir = if ($MyInvocation.MyCommand.Path) {
    Split-Path -Parent $MyInvocation.MyCommand.Path
} else {
    Get-Location
}
$ProjectRoot = Split-Path -Parent $BackendDir
$FrontendDir = Join-Path $ProjectRoot 'sequence_frontend'
$VenvPython = Join-Path $BackendDir '.venv\Scripts\python.exe'
$SetupScript = Join-Path $BackendDir 'setup_windows.ps1'

Write-Host "=======================================================" -ForegroundColor Cyan
Write-Host "    Starting Sequence Detection Development Mode       " -ForegroundColor Cyan
Write-Host "=======================================================" -ForegroundColor Cyan
Write-Host ""

# ------------------------------------------------------------------------------
# 0. Kill any lingering dev servers from previous runs (ports 8000 & 5173, etc.)
# ------------------------------------------------------------------------------
function Stop-LingeringDevServers {
    param(
        [int]$BackendPort = 8000,
        [int]$FrontendPort = 5173
    )
    Write-Host "[CLEANUP] Stopping any lingering servers or background workers..." -ForegroundColor Cyan

    # 1. Kill any process listening on backend port
    try {
        $pids = Get-NetTCPConnection -LocalPort $BackendPort -State Listen -ErrorAction SilentlyContinue | Select-Object -ExpandProperty OwningProcess -Unique
        foreach ($pidToKill in $pids) {
            if ($pidToKill -and $pidToKill -gt 0 -and $pidToKill -ne $PID) {
                Write-Host "  Freeing backend port $BackendPort (terminating PID $pidToKill)..." -ForegroundColor Yellow
                Stop-Process -Id $pidToKill -Force -ErrorAction SilentlyContinue
            }
        }
    } catch {}

    # 2. Kill any process listening on frontend port
    try {
        $pids5173 = Get-NetTCPConnection -LocalPort $FrontendPort -State Listen -ErrorAction SilentlyContinue | Select-Object -ExpandProperty OwningProcess -Unique
        foreach ($pidToKill in $pids5173) {
            if ($pidToKill -and $pidToKill -gt 0 -and $pidToKill -ne $PID) {
                Write-Host "  Freeing frontend port $FrontendPort (terminating PID $pidToKill)..." -ForegroundColor Yellow
                Stop-Process -Id $pidToKill -Force -ErrorAction SilentlyContinue
            }
        }
    } catch {}

    # 3. Kill any Python / Uvicorn process running from this backend's .venv or running app.main:app
    try {
        $CimProcs = Get-CimInstance Win32_Process -Filter "Name like 'python%.exe' or Name='uvicorn.exe'" -ErrorAction SilentlyContinue
        foreach ($lp in $CimProcs) {
            if ($lp.ProcessId -ne $PID -and (($lp.ExecutablePath -like "*$BackendDir*") -or ($lp.CommandLine -like "*$BackendDir*") -or ($lp.CommandLine -like "*app.main:app*"))) {
                Write-Host "  Stopping previous backend Python process (PID $($lp.ProcessId))..." -ForegroundColor Yellow
                Stop-Process -Id $lp.ProcessId -Force -ErrorAction SilentlyContinue
            }
        }
    } catch {}

    # 4. Close previous console windows
    try {
        $CmdProcs = Get-CimInstance Win32_Process -Filter "Name='cmd.exe'" -ErrorAction SilentlyContinue
        foreach ($cp in $CmdProcs) {
            if ($cp.ProcessId -ne $PID -and (($cp.CommandLine -like "*Sequence Detection - Backend API*") -or ($cp.CommandLine -like "*Sequence Detection - Frontend UI*"))) {
                Write-Host "  Closing previous console window (PID $($cp.ProcessId))..." -ForegroundColor Yellow
                Stop-Process -Id $cp.ProcessId -Force -ErrorAction SilentlyContinue
            }
        }
    } catch {}

    Start-Sleep -Milliseconds 600
}

# Determine target backend port and clean up lingering processes
$TargetPort = if ($env:BACKEND_PORT) { [int]$env:BACKEND_PORT } elseif ($env:PORT) { [int]$env:PORT } else { 8000 }
Stop-LingeringDevServers -BackendPort $TargetPort -FrontendPort 5173

# ------------------------------------------------------------------------------
# 1. Verify virtual environment & pre-flight core dependencies
# ------------------------------------------------------------------------------
$VenvHealthy = $false
if (Test-Path $VenvPython) {
    try {
        $check = & $VenvPython -c "
import sys
try:
    import fastapi, uvicorn, cv2, torch, torchvision, yaml, ultralytics, mediapipe, timm, safetensors, huggingface_hub, scipy
    print('VENV_HEALTHY')
except Exception:
    sys.exit(1)
" 2>&1
        if ($LASTEXITCODE -eq 0 -and ($check -match 'VENV_HEALTHY')) {
            $VenvHealthy = $true
        }
    } catch {
        $VenvHealthy = $false
    }
}

if (-not $VenvHealthy) {
    Write-Host "[PRE-FLIGHT] Virtual environment is missing or incomplete (setup was interrupted). Running automated setup..." -ForegroundColor Yellow
    & powershell -NoProfile -ExecutionPolicy Bypass -File $SetupScript
    if ($LASTEXITCODE -ne 0 -or -not (Test-Path $VenvPython)) {
        throw "Automated setup failed. Please review logs above."
    }
}

# ------------------------------------------------------------------------------
# 2. Fast pre-flight check: install bundled TrayCount wheel if missing
# ------------------------------------------------------------------------------
$TrayCountOk = try {
    $tcOut = & $VenvPython -c "
import sys
try:
    from traycount import TrayCounter
    print('TRAYCOUNT_OK')
except Exception:
    sys.exit(1)
" 2>&1
    ($LASTEXITCODE -eq 0 -and ($tcOut -match 'TRAYCOUNT_OK'))
} catch { $false }

if (-not $TrayCountOk) {
    $WhlDir = Join-Path $BackendDir 'app\ml\whl'
    $SearchRoot = if (Test-Path $WhlDir) { $WhlDir } else { Join-Path $BackendDir 'app\ml' }
    $TrayCountWheel = Get-ChildItem -Path $SearchRoot -Filter 'traycount-*.whl' -Recurse -ErrorAction SilentlyContinue |
        Sort-Object {
            if ($_.Name -match 'traycount-([0-9]+(\.[0-9]+)*)') {
                try { [System.Version]$matches[1] } catch { [System.Version]'0.0.0' }
            } else {
                [System.Version]'0.0.0'
            }
        } -Descending | Select-Object -First 1

    if ($TrayCountWheel) {
        Write-Host "[PRE-FLIGHT] Installing $($TrayCountWheel.Name)..." -ForegroundColor Yellow
        & $VenvPython -m pip install --quiet --no-deps --force-reinstall $TrayCountWheel.FullName
        if ($LASTEXITCODE -ne 0) {
            Write-Host "[WARN] Failed to install $($TrayCountWheel.Name)." -ForegroundColor Yellow
        }
    } else {
        Write-Host "[WARN] traycount wheel not found in app\ml\whl." -ForegroundColor Yellow
    }
}

# ------------------------------------------------------------------------------
# 3. Verify CUDA kernels when an NVIDIA GPU is present; auto-repair via setup if broken
# ------------------------------------------------------------------------------
$HasNvidia = $false
$SmiCmd = Get-Command nvidia-smi -ErrorAction SilentlyContinue
if ($SmiCmd) {
    try {
        & $SmiCmd.Source --query-gpu=name --format=csv,noheader 2>$null | Out-Null
        if ($LASTEXITCODE -eq 0) { $HasNvidia = $true }
    } catch {}
}

if ($HasNvidia) {
    $CudaOk = try {
        $cOut = & $VenvPython -c "
import sys
try:
    import torch
    if torch.cuda.is_available() and torch.cuda.device_count() > 0:
        x = torch.ones((1,), device='cuda:0')
        assert (x + 1).item() == 2
        torch.cuda.synchronize(0)
        print('GPU_OK:' + torch.cuda.get_device_name(0))
    else:
        sys.exit(1)
except Exception:
    sys.exit(1)
" 2>&1
        if ($LASTEXITCODE -eq 0 -and ($cOut -match 'GPU_OK:(.+)')) {
            $matches[1].Trim()
        } else {
            $null
        }
    } catch { $null }

    if ($CudaOk) {
        Write-Host "[PRE-FLIGHT] GPU Ready: $CudaOk" -ForegroundColor Green
    } else {
        Write-Host "[PRE-FLIGHT] NVIDIA GPU detected, but installed PyTorch cannot execute CUDA kernels. Running setup..." -ForegroundColor Yellow
        & powershell -NoProfile -ExecutionPolicy Bypass -File $SetupScript
        if ($LASTEXITCODE -ne 0) {
            throw "Failed to repair CUDA PyTorch environment."
        }
    }
} else {
    Write-Host "[PRE-FLIGHT] Inference device: CPU (no NVIDIA GPU detected)" -ForegroundColor Yellow
}

# ------------------------------------------------------------------------------
# 4. Ensure frontend dependencies exist
# ------------------------------------------------------------------------------
$ViteBin = Join-Path $FrontendDir 'node_modules\.bin\vite.cmd'
if (-not (Test-Path $ViteBin)) {
    Write-Host "[PRE-FLIGHT] Frontend dependencies missing. Installing node_modules..." -ForegroundColor Yellow
    Push-Location $FrontendDir
    try {
        & npm.cmd install --include=optional
        if ($LASTEXITCODE -ne 0) {
            & npm.cmd install --include=optional --ignore-scripts
            if ($LASTEXITCODE -ne 0) { throw "npm install failed in $FrontendDir." }
        }
    } finally {
        Pop-Location
    }
}

# ------------------------------------------------------------------------------
# 5. Auto-select an available backend port (default 8000)
# ------------------------------------------------------------------------------
$Port = $TargetPort

function Test-PortFree([int]$CheckPort) {
    $res = try {
        & $VenvPython -c "
import socket, sys
s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
rc = 0
try:
    s.bind(('0.0.0.0', int(sys.argv[1])))
except OSError:
    rc = 1
finally:
    s.close()
sys.exit(rc)
" $CheckPort 2>&1
        ($LASTEXITCODE -eq 0)
    } catch { $false }
    return $res
}

while (-not (Test-PortFree $Port)) {
    Write-Host "Port $Port is busy. Trying port $($Port + 1)..." -ForegroundColor Yellow
    $Port++
    if ($Port -gt ($TargetPort + 50)) {
        throw "Could not find an open port in range $TargetPort-$Port."
    }
}
if ($Port -ne $TargetPort) {
    Write-Host "Note: Port $TargetPort was busy. Using available port $Port for backend." -ForegroundColor Cyan
}

# ------------------------------------------------------------------------------
# 6. Start Backend and Frontend in dedicated console windows
# ------------------------------------------------------------------------------
Write-Host ""
Write-Host "[1/2] Starting Sequence Detection Backend API (Port $Port)..." -ForegroundColor Cyan
$BackendCmd = "`"$VenvPython`" -m uvicorn app.main:app --reload --reload-dir app --reload-exclude .venv* --reload-exclude data* --reload-exclude recordings* --port $Port"
Start-Process -FilePath "cmd.exe" -ArgumentList "/k", "title Sequence Detection - Backend API && $BackendCmd" -WorkingDirectory $BackendDir

Write-Host "[2/2] Starting Sequence Detection Frontend UI (Port 5173)..." -ForegroundColor Cyan
$FrontendCmd = "set VITE_API_BASE_URL=http://localhost:$Port && npm run dev"
Start-Process -FilePath "cmd.exe" -ArgumentList "/k", "title Sequence Detection - Frontend UI && $FrontendCmd" -WorkingDirectory $FrontendDir

Write-Host ""
Write-Host "=======================================================" -ForegroundColor Green
Write-Host "[SUCCESS] Sequence Detection development servers started!" -ForegroundColor Green
Write-Host ""
Write-Host "  Frontend Application : http://localhost:5173" -ForegroundColor White
Write-Host "  Backend Health Check : http://localhost:$Port/health" -ForegroundColor White
Write-Host "  API Documentation    : http://localhost:$Port/docs" -ForegroundColor White
Write-Host ""
Write-Host "Keep both console windows open while working." -ForegroundColor Gray
Write-Host "To stop all servers at any time, simply double-click stop.bat" -ForegroundColor Gray
Write-Host "=======================================================" -ForegroundColor Green
