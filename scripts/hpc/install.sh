#!/bin/bash
# scripts/hpc/install.sh — Run once on a login node (donphan) to set up the
# project's Python virtual environment using the official vsc-venv wrapper.
#
# Usage (from project root):
#   bash scripts/hpc/install.sh
#
# Prerequisites:
#   - Connected to VSC via the web portal (HPC Login → Interactive Apps > Shell tmux)
#   - Set cluster to "donphan (interactive/debug)"
#   - Project cloned into $VSC_HOME (e.g. git clone <repo> && cd <repo>)

set -euo pipefail

# ---------------------------------------------------------------------------
# Redirect caches away from $VSC_HOME (very limited at ~3 GB) to scratch.
# This must be done before any pip/module activity.
# ---------------------------------------------------------------------------
export XDG_CACHE_HOME="$VSC_SCRATCH/.cache"
export UV_CACHE_DIR="$VSC_SCRATCH/.cache/uv"
export MPLCONFIGDIR="$VSC_SCRATCH/.config/matplotlib"
mkdir -p "$XDG_CACHE_HOME" "$UV_CACHE_DIR" "$MPLCONFIGDIR"

echo "==> Loading vsc-venv module..."
module load vsc-venv

echo "==> Activating virtual environment (created per-cluster in \$VSC_DATA)..."
# vsc-venv transparently creates and manages a per-cluster venv in $VSC_DATA,
# loading the modules listed in env/hpc/modules.txt and pip-installing anything
# in env/hpc/requirements.txt that is not already satisfied.
source vsc-venv --activate \
    --modules env/hpc/modules.txt \
    --requirements env/hpc/requirements.txt

echo "==> Registering Jupyter kernel..."
python -m ipykernel install --user --name="sel3_${VSC_INSTITUTE_CLUSTER}" \
    --display-name "SEL3 (${VSC_INSTITUTE_CLUSTER})"

echo ""
echo "✅ Environment ready. Kernel: sel3_${VSC_INSTITUTE_CLUSTER}"
echo "   To activate in future sessions:"
echo "     module load vsc-venv && source vsc-venv --activate --modules env/hpc/modules.txt"
echo ""
echo "   Remember to also set cache dirs in any interactive session:"
echo "     export XDG_CACHE_HOME=\$VSC_SCRATCH/.cache"
