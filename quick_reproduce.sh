#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"

ENV=spare-repro
if ! command -v conda >/dev/null 2>&1; then
    for candidate in "${CONDA_INSTALL_PREFIX:-}" "$HOME/miniforge3" "$HOME/miniconda3" "$HOME/anaconda3" \
                     /opt/conda; do
        if [ -n "$candidate" ] && [ -x "$candidate/bin/conda" ]; then
            export PATH="$candidate/bin:$PATH"
            break
        fi
    done
fi
if ! command -v conda >/dev/null 2>&1; then
    echo "conda not found -- install it with one command first:"
    echo "  bash install_conda.sh"
    exit 1
fi
eval "$(conda shell.bash hook)"

if ! conda env list | awk '{print $1}' | grep -qx "$ENV"; then
    echo "[quick_reproduce] creating conda env '$ENV' (python 3.10) ..."
    conda create -y -n "$ENV" python=3.10 -c conda-forge --override-channels
fi
echo "[quick_reproduce] installing cpu-only deps (torch/numpy/pyyaml) ..."
conda run -n "$ENV" --no-capture-output python -m pip install -q \
    torch --index-url https://download.pytorch.org/whl/cpu
conda run -n "$ENV" --no-capture-output python -m pip install -q numpy pyyaml

if [ "${1:-}" = "--setup-only" ]; then
    echo "[quick_reproduce] environment '$ENV' ready."
    exit 0
fi

QUICK_PY="replay/quick_reproduce.py"; [ -f "$QUICK_PY" ] || QUICK_PY="quick_reproduce.py"
conda run -n "$ENV" --no-capture-output python "$QUICK_PY"
