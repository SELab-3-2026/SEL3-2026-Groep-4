#!/bin/bash
# scripts/hpc/install.sh — Run once on a login node (donphan) to set up the
# project's Python virtual environment using the official vsc-venv wrapper.
#
# Usage (from project root):
#   bash scripts/hpc/install.sh
#
# Prerequisites:
#   - Connected to VSC via the web portal (HPC Login → Shell tmux)
#   - Project cloned to $VSC_DATA or $VSC_HOME
#   - Run from the project root directory

set -euo pipefail

module load vsc-venv

source vsc-venv --activate \
    --modules env/hpc/modules.txt \
    --requirements env/hpc/requirements.txt

python -m ipykernel install --user --name="sel3_${VSC_INSTITUTE_CLUSTER}" \
    --display-name "SEL3 (${VSC_INSTITUTE_CLUSTER})"

echo ""
echo "✅ Environment ready. Kernel: sel3_${VSC_INSTITUTE_CLUSTER}"
echo "   Activate in future sessions with:"
echo "     module load vsc-venv && source vsc-venv --activate --modules env/hpc/modules.txt"
