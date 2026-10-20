#!/usr/bin/env bash
#
# Startup script for TagGUI (llama.cpp edition).
#
# On first use it creates a virtual environment in .venv and installs the
# dependencies from requirements.txt into it. On subsequent runs it detects
# the existing virtual environment and just starts the GUI.
#
# Usage:
#   ./run.sh                Start TagGUI (setting up the venv on first use).
#   ./run.sh --reinstall    Delete the venv, recreate it, and start TagGUI.
#                           Use this after pulling changes that modified
#                           requirements.txt.
#
# Note: LLM inference is not done by this application. Set the absolute path
# to your local llama.cpp `llama-server` binary in File -> Settings, for
# example /home/user/AI/llama.cpp/build/bin/llama-server

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

VENV_DIR="$SCRIPT_DIR/.venv"
VENV_PYTHON="$VENV_DIR/bin/python"
REQUIREMENTS_FILE="$SCRIPT_DIR/requirements.txt"
GUI_SCRIPT="$SCRIPT_DIR/taggui/run_gui.py"
PYTHON_BIN="${PYTHON_BIN:-python3}"

create_venv_and_install_dependencies() {
    if ! command -v "$PYTHON_BIN" > /dev/null 2>&1; then
        echo "Error: $PYTHON_BIN was not found. Please install Python 3.11 " \
             "or newer (including the python3-venv package on Debian/Ubuntu)" \
             "and try again." >&2
        exit 1
    fi
    echo "Creating a virtual environment in $VENV_DIR ..."
    "$PYTHON_BIN" -m venv "$VENV_DIR"
    echo "Installing dependencies from $REQUIREMENTS_FILE ..."
    "$VENV_PYTHON" -m pip install --upgrade pip
    "$VENV_PYTHON" -m pip install -r "$REQUIREMENTS_FILE"
    echo "Dependencies installed."
}

# Handle the optional --reinstall flag.
if [ "${1:-}" = "--reinstall" ]; then
    shift
    echo "Removing the existing virtual environment ..."
    rm -rf "$VENV_DIR"
fi

if [ -x "$VENV_PYTHON" ]; then
    echo "Virtual environment found. Starting TagGUI..."
else
    echo "Virtual environment not found. This appears to be the first run."
    create_venv_and_install_dependencies
    echo "Starting TagGUI..."
fi

exec "$VENV_PYTHON" "$GUI_SCRIPT" "$@"
