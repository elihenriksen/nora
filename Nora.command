#!/bin/bash
# Double-click this file in Finder to launch Nora.
# It opens in Terminal, activates the project's virtualenv, and starts the
# interactive launcher menu.
#
# Linux/Windows users: there's no equivalent double-click launcher; run
#   python -m nora              # interactive launcher
#   python -m nora.cli --help   # plain CLI
#
# Debug mode:
#   ./Nora.command --debug         # turns on NORA_DEBUG for this session
#   NORA_DEBUG=1 ./Nora.command    # equivalent — env vars are forwarded
#
# All extra args are passed through to `python -m nora`.

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
  exit 1
fi

# Check 3: .env file exists (gives a friendlier error than the launcher's)
if [ ! -f "$HERE/.env" ]; then
  echo "No .env file found at $HERE/.env"
  echo
  echo "Create one with:"
  echo "  cp .env.example .env"
  echo "  \$EDITOR .env       # add your API key(s)"
  echo
  read -n 1 -s -r -p "Press any key to close..."
  exit 1
fi

clear
# `exec` replaces the shell, so all env vars (including NORA_DEBUG) and any
# args passed to this script ($@) flow through to the launcher.
exec "$PY" -m nora "$@"
