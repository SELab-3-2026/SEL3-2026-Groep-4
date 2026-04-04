#!/bin/bash -l
# scripts/hpc/install.sh
#
# Usage (on any compute node):
#   bash scripts/hpc/install.sh
#
# Batch usage:
#   qsub scripts/hpc/install.sh

#PBS -N brittlestar-install
#PBS -l walltime=01:00:00

set -euo pipefail

# Preliminary status echo
echo ">>> Starting installation job $PBS_JOBID on $(hostname)..."

if [ -n "$PBS_O_WORKDIR" ]; then
    cd "$PBS_O_WORKDIR"
fi

# Mirror configs to $VSC_DATA to avoid home quota limits (3GB)
# vsc-venv manages environments relative to the requirements file
PROJ_NAME=$(basename "$PWD")
HPC_CONFIG_DIR="$VSC_DATA/$PROJ_NAME/env/hpc"
mkdir -p "$HPC_CONFIG_DIR"
cp env/hpc/*.txt "$HPC_CONFIG_DIR/"

# Keep caches off $VSC_HOME (quota ~3 GB).
export PIP_CACHE_DIR="$VSC_SCRATCH/.cache/pip"
export UV_CACHE_DIR="$VSC_SCRATCH/.cache/uv"
mkdir -p "$PIP_CACHE_DIR" "$UV_CACHE_DIR"

module load vsc-venv

echo ">>> Synchronizing and activating environment (vsc-venv)..."
set +euo pipefail
source vsc-venv --activate \
    --modules "$HPC_CONFIG_DIR/modules.txt" \
    --requirements "$HPC_CONFIG_DIR/requirements.txt"
set -euo pipefail

# Prepend venv to PYTHONPATH to shadow system packages (e.g. SciPy-bundle compiled
# against NumPy 1.x) that are injected by the loaded modules.
VENV_LIB_DIR="$VIRTUAL_ENV/lib/python$(python -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')/site-packages"
export PYTHONPATH="$VENV_LIB_DIR:${PYTHONPATH:-}"

echo '>>> Installing ipykernel...'
CLUSTER_ID="${VSC_INSTITUTE_CLUSTER:-generic}"
python -m ipykernel install --user --name="sel3_${CLUSTER_ID}" \
    --display-name "SEL3 (${CLUSTER_ID})"

echo '>>> Done'
