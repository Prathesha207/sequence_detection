#!/usr/bin/env bash
# Stop any running Sequence Detection dev servers on Linux / Ubuntu.
set -euo pipefail

BACKEND_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TARGET_PORT="${BACKEND_PORT:-${PORT:-8000}}"
FRONTEND_PORT=5173

echo "======================================================="
echo "      Stopping Sequence Detection Servers (Linux)...   "
echo "======================================================="

kill_port() {
  local p="$1"
  if command -v fuser >/dev/null 2>&1; then
    fuser -k "${p}/tcp" 2>/dev/null || true
  elif command -v lsof >/dev/null 2>&1; then
    lsof -ti:"${p}" | xargs kill -9 2>/dev/null || true
  fi
}

echo "Freeing port $TARGET_PORT and port $FRONTEND_PORT..."
kill_port "$TARGET_PORT"
kill_port "$FRONTEND_PORT"

echo "Terminating any lingering backend and frontend workers..."
pkill -f "$BACKEND_DIR/.venv/bin/python -m uvicorn app.main:app" 2>/dev/null || true
pkill -f "uvicorn app.main:app" 2>/dev/null || true
pkill -f "vite --host 0.0.0.0" 2>/dev/null || true

echo ""
echo "[SUCCESS] Stopped all Sequence Detection server processes."
echo "======================================================="
