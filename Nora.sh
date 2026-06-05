#!/usr/bin/env bash
# Nora launcher for Linux.
# Run from the project root:
#   ./Nora.sh                 normal launch
#   ./Nora.sh --debug         with NORA_DEBUG=1 propagated to subprocesses
#
# macOS users: use Nora.command. Windows users: use Nora.bat.
# The cross-platform fallback is `python -m nora`.

set -e
HERE="$(cd "$(dirname "$0")" && pwd)"
cd "$HERE"

PY="$HERE/.venv/bin/python"

# Check 1: virtualenv exists
if [ ! -x "$PY" ]; then
  echo "No virtualenv found at $HERE/.venv"
  echo
  echo "Set it up with:"
  echo "  python3 -m venv .venv"
  echo "  .venv/bin/pip install -e ."
  echo
  read -n 1 -s -r -p "Press any key to close..."
  echo
  exit 1
fi

# Check 2: nora package is actually installed in the venv
if ! "$PY" -c "import nora" 2>/dev/null; then
  echo "The 'nora' package isn't installed in $HERE/.venv"
  echo
  echo "Install it with:"
  echo "  .venv/bin/pip install -e ."
  echo
  read -n 1 -s -r -p "Press any key to close..."
  echo
  exit 1
fi

# Check 3: .env file exists
if [ ! -f "$HERE/.env" ]; then
  echo "No .env file found at $HERE/.env"
  echo
  echo "Create one with:"
  echo "  cp .env.example .env"
  echo "  \$EDITOR .env"
  echo
  read -n 1 -s -r -p "Press any key to close..."
  echo
  exit 1
fi

clear
exec "$PY" -m nora "$@"
