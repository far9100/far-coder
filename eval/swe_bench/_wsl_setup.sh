#!/usr/bin/env bash
# Idempotent setup script for the WSL scoring venv. Safe to re-run.
set -e
VENV="$HOME/.farcode-swe-venv"

echo "[setup] HOME=$HOME"
echo "[setup] VENV=$VENV"
echo "[setup] CWD=$PWD"

if [ ! -d "$VENV" ]; then
    echo "[setup] creating venv at $VENV"
    python3 -m venv "$VENV"
fi

source "$VENV/bin/activate"
echo "[setup] which python: $(which python)"
echo "[setup] python version: $(python --version)"

echo "[setup] upgrading pip"
pip install -q --upgrade pip

echo "[setup] installing swebench + datasets (may take 1-3 min)"
pip install -q "datasets>=2.18" "swebench>=2.1"

echo "[setup] verify swebench import"
python -c "import swebench; print('swebench', swebench.__version__, 'OK')"

echo "[setup] verify docker"
docker version --format '{{.Server.Version}}' || echo "DOCKER NOT REACHABLE — enable WSL integration in Docker Desktop"

echo "[setup] DONE"
