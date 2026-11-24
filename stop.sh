#!/usr/bin/env bash
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
cd "$SCRIPT_DIR/sequence_backend"
chmod +x stop_dev_linux.sh
./stop_dev_linux.sh "$@"
