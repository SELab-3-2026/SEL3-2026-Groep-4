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

module load vsc-venv

echo ">>> Synchronizing and activating environment (vsc-venv)..."
set +euo pipefail
source vsc-venv --activate \
    --modules "$HPC_CONFIG_DIR/modules.txt" \
    --requirements "$HPC_CONFIG_DIR/requirements.txt"
set -euo pipefail

# Force the venv path to the front of PYTHONPATH to override system modules (e.g. NumPy 1.2x)
VENV_LIB_DIR="$VIRTUAL_ENV/lib/python$(python -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')/site-packages"
export PYTHONPATH="$VENV_LIB_DIR:$PYTHONPATH"

echo '>>> Installing ipykernel...'
CLUSTER_ID="${VSC_INSTITUTE_CLUSTER:-generic}"
python -m ipykernel install --user --name="sel3_${CLUSTER_ID}" \
    --display-name "SEL3 (${CLUSTER_ID})"

echo '>>> Done'
