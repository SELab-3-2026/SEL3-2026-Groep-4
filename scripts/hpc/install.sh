#!/bin/bash
# scripts/hpc/install.sh
#
# Usage:
#   module swap cluster/donphan
#   qsub -I -l nodes=1:ppn=8:gpus=1
#   (Wait for session to start, then:)
#   bash scripts/hpc/install.sh

set -eo pipefail

# Keep caches off $VSC_HOME (quota ~3 GB).
export PIP_CACHE_DIR="$VSC_SCRATCH/.cache/pip"
export UV_CACHE_DIR="$VSC_SCRATCH/.cache/uv"
mkdir -p "$PIP_CACHE_DIR" "$UV_CACHE_DIR"

# Ensure we are in the project root
if [ -n "$PBS_O_WORKDIR" ]; then
    cd "$PBS_O_WORKDIR"
fi

module load vsc-venv

echo 'Synchronizing environment...'
# Step 1: Sync (build/update) the environment
vsc-venv \
    --modules env/hpc/modules.txt \
    --requirements env/hpc/requirements.txt

echo 'Activating environment...'
# Step 2: Activate the environment
source vsc-venv --activate \
    --modules env/hpc/modules.txt \
    --requirements env/hpc/requirements.txt

# Step 3: Force upgrade shared system dependencies to ensure venv precedence
echo 'Applying library overlays (NumPy, Protobuf)...'
pip install --upgrade --no-deps numpy protobuf

echo 'Installing ipykernel...'
python -m ipykernel install --user --name="sel3_${VSC_INSTITUTE_CLUSTER}" \
    --display-name "SEL3 (${VSC_INSTITUTE_CLUSTER})"

echo 'Done'

