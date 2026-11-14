#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
bash quick_reproduce.sh --setup-only
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
eval "$(conda shell.bash hook)"
ROBOTCAR_PY="replay/robotcar_reproduce.py"; [ -f "$ROBOTCAR_PY" ] || ROBOTCAR_PY="robotcar_reproduce.py"
conda run -n "$ENV" --no-capture-output python "$ROBOTCAR_PY"
