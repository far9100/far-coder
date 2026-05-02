#!/usr/bin/env bash
# Run inside a freshly-installed WSL Ubuntu to prepare the SWE-bench
# scoring environment. Usage:
#
#     # On Windows:
#     wsl --install -d Ubuntu --no-launch
#     wsl -d Ubuntu                         # first launch — set username/password
#     # Inside Ubuntu, from the project dir mounted under /mnt/c/...:
#     bash eval/swe_bench/setup_wsl.sh
#
# This script:
# 1. Installs system deps (Python 3.11, Docker CLI, git)
# 2. Sets up a Python venv and installs `swebench`
# 3. Verifies Docker Desktop's Linux engine is reachable from WSL
#
# It does NOT run any evaluation — that's done after with score.py.
set -euo pipefail

PROJECT="${PROJECT:-/mnt/c/Users/fartw/OneDrive/Desktop/github/code/ai-coder}"
VENV="${VENV:-$HOME/.farcode-swe-venv}"

echo "[setup] project = $PROJECT"
echo "[setup] venv    = $VENV"

if [ ! -d "$PROJECT" ]; then
    echo "ERROR: project dir not found at $PROJECT"
    echo "Set PROJECT=<path> and re-run."
    exit 1
fi

echo "[setup] step 1: apt deps"
sudo apt-get update -qq
sudo apt-get install -y -qq python3 python3-venv python3-pip git docker.io || \
    sudo apt-get install -y -qq python3 python3-venv python3-pip git

echo "[setup] step 2: python venv"
if [ ! -d "$VENV" ]; then
    python3 -m venv "$VENV"
fi
source "$VENV/bin/activate"
pip install -q --upgrade pip
pip install -q "datasets>=2.18" "swebench>=2.1"

echo "[setup] step 3: verify swebench imports"
python -c "import swebench; print('swebench', swebench.__version__, 'OK')"

echo "[setup] step 4: verify Docker reachable from WSL"
if docker version >/dev/null 2>&1; then
    echo "[setup] docker OK"
else
    echo "[setup] WARNING: Docker not reachable from WSL."
    echo "  In Docker Desktop -> Settings -> Resources -> WSL integration,"
    echo "  enable 'Ubuntu' (or whatever distro you installed)."
fi

echo
echo "==========================================================="
echo "Setup complete. To score predictions:"
echo "    cd $PROJECT"
echo "    source $VENV/bin/activate"
echo "    python -m eval.swe_bench.score \\"
echo "        --predictions out/cmp-full.jsonl \\"
echo "        --dataset princeton-nlp/SWE-bench_Lite \\"
echo "        --run-id cmp-full"
echo "    python -m eval.swe_bench.score \\"
echo "        --predictions out/cmp-bare.jsonl \\"
echo "        --dataset princeton-nlp/SWE-bench_Lite \\"
echo "        --run-id cmp-bare"
echo "==========================================================="
