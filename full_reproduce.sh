#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"

ENV=spare-full
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
    echo "[full_reproduce] creating conda env '$ENV' (python 3.10) ..."
    conda create -y -n "$ENV" python=3.10 -c conda-forge --override-channels
fi
echo "[full_reproduce] installing deps (torch with CUDA support, numpy, pyyaml) ..."
conda run -n "$ENV" --no-capture-output python -m pip install -q torch numpy pyyaml

if ! printf '%s\n' "$@" | grep -qx -- --cpu; then
    if ! conda run -n "$ENV" python -c "import torch,sys; sys.exit(0 if torch.cuda.is_available() else 1)" >/dev/null 2>&1; then
        echo "[full_reproduce] no CUDA GPU detected -- falling back to --cpu (measured: ~15-30 min on a modern multi-core CPU)"
        export PAPER_REPLAY_TIMEOUT_S="${PAPER_REPLAY_TIMEOUT_S:-86400}"
        set -- "$@" --cpu
    fi
fi

RUN_PY="replay/run_paper_tables.py"; [ -f "$RUN_PY" ] || RUN_PY="run_paper_tables.py"
conda run -n "$ENV" --no-capture-output python "$RUN_PY" "$@"
