#!/usr/bin/env bash
# Start backend and frontend together for Linux development.
set -euo pipefail

BACKEND_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
FRONTEND_DIR="$(cd "$BACKEND_DIR/../sequence_frontend" && pwd)"

# Configurable base port (default 8000, or override with BACKEND_PORT=8080)
TARGET_PORT="${BACKEND_PORT:-${PORT:-8000}}"
FRONTEND_PORT=5173

# Helper to kill any process holding a specific port
kill_port() {
  local p="$1"
  if command -v fuser >/dev/null 2>&1; then
    fuser -k "${p}/tcp" 2>/dev/null || true
  elif command -v lsof >/dev/null 2>&1; then
    lsof -ti:"${p}" | xargs kill -9 2>/dev/null || true
  fi
}

# 0. Clean up any lingering dev server processes from previous sessions
echo "[CLEANUP] Stopping any lingering backend (port $TARGET_PORT) or frontend (port $FRONTEND_PORT) processes..."
kill_port "$TARGET_PORT"
kill_port "$FRONTEND_PORT"
pkill -f "$BACKEND_DIR/.venv/bin/python -m uvicorn app.main:app" 2>/dev/null || true
pkill -f "uvicorn app.main:app" 2>/dev/null || true
pkill -f "vite --host 0.0.0.0" 2>/dev/null || true
sleep 0.5

# 1. Verify virtual environment & pre-flight core dependencies
VENV_PYTHON="$BACKEND_DIR/.venv/bin/python"
VENV_HEALTHY=0
if [[ -x "$VENV_PYTHON" ]]; then
  if "$VENV_PYTHON" -c "
import sys
try:
    import fastapi, uvicorn, cv2, torch, torchvision, yaml, ultralytics, mediapipe, timm, safetensors, huggingface_hub, scipy
    print('VENV_HEALTHY')
except Exception:
    sys.exit(1)
" 2>/dev/null | grep -q "VENV_HEALTHY"; then
    VENV_HEALTHY=1
  fi
fi

if [[ "$VENV_HEALTHY" -ne 1 ]]; then
  echo "Backend environment is missing or incomplete (setup was interrupted). Running setup_linux.sh..."
  bash "$BACKEND_DIR/setup_linux.sh"
fi

# 2. Fast pre-flight check: install bundled TrayCount wheel if missing
if ! "$VENV_PYTHON" -c "from traycount import TrayCounter" 2>/dev/null; then
  WHL_FILE="$(find "$BACKEND_DIR/app/ml/whl" "$BACKEND_DIR/app/ml" -maxdepth 2 -name 'traycount-*.whl' -print 2>/dev/null | sort -V -r | head -n 1 || true)"
  if [[ -n "$WHL_FILE" && -f "$WHL_FILE" ]]; then
    echo "Installing $(basename "$WHL_FILE")..."
    "$VENV_PYTHON" -m pip install --quiet --no-deps --force-reinstall "$WHL_FILE" 2>/dev/null || true
  fi
fi

# 3. Repair an incompatible CUDA wheel automatically
if command -v nvidia-smi >/dev/null 2>&1 && \
   ! "$VENV_PYTHON" -c "import torch; x=torch.ones((1,), device='cuda:0'); assert (x+1).item()==2; torch.cuda.synchronize(0)" 2>/dev/null; then
  echo "NVIDIA GPU detected, but installed PyTorch cannot execute its CUDA kernels. Running setup..."
  bash "$BACKEND_DIR/setup_linux.sh"
fi

# 4. Ensure frontend dependencies exist
if [[ ! -x "$FRONTEND_DIR/node_modules/.bin/vite" ]]; then
  echo "Frontend dependencies are missing. Installing node_modules..."
  cd "$FRONTEND_DIR" && (npm install --include=optional || npm install --include=optional --ignore-scripts)
  cd "$BACKEND_DIR"
fi

# 5. Function to check if a port is available
is_port_free() {
  "$VENV_PYTHON" -c "
import socket, sys
s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
try:
    s.bind(('0.0.0.0', int(sys.argv[1])))
    s.close()
    sys.exit(0)
except OSError:
    sys.exit(1)
" "$1"
}

# Auto-select an available port if the target port is already in use by an external service
PORT="$TARGET_PORT"
while ! is_port_free "$PORT"; do
  echo "Port $PORT is already in use by another application. Trying port $((PORT + 1))..."
  PORT=$((PORT + 1))
  if [[ "$PORT" -gt "$((TARGET_PORT + 50))" ]]; then
    echo "ERROR: Could not find an open port in range $TARGET_PORT-$PORT."
    exit 1
  fi
done

if [[ "$PORT" != "$TARGET_PORT" ]]; then
  echo "Note: Port $TARGET_PORT was busy. Using available port $PORT for backend."
fi

# 6. Cleanup handler for graceful shutdown
cleanup() {
  echo ""
  echo "Stopping development servers..."
  if [[ -n "${BACKEND_PID:-}" ]]; then
    pkill -P "$BACKEND_PID" 2>/dev/null || true
    kill "$BACKEND_PID" 2>/dev/null || true
  fi
  if [[ -n "${FRONTEND_PID:-}" ]]; then
    pkill -P "$FRONTEND_PID" 2>/dev/null || true
    kill "$FRONTEND_PID" 2>/dev/null || true
  fi
  kill_port "$PORT"
  kill_port "$FRONTEND_PORT"
}
trap cleanup EXIT INT TERM

cd "$BACKEND_DIR"
"$VENV_PYTHON" -m uvicorn app.main:app --reload --reload-dir app --reload-exclude ".venv*" --reload-exclude "data*" --reload-exclude "recordings*" --port "$PORT" &
BACKEND_PID=$!

cd "$FRONTEND_DIR"
VITE_API_BASE_URL="http://localhost:$PORT" npm run dev -- --host 0.0.0.0 &
FRONTEND_PID=$!

echo ""
echo "=================================================="
echo " Sequence Detection Dev Servers Started"
echo " Backend:  http://localhost:$PORT"
echo " Frontend: http://localhost:5173"
echo " Docs:     http://localhost:$PORT/docs"
echo "=================================================="
echo ""
echo "Press Ctrl+C to stop all servers."
echo ""

wait
