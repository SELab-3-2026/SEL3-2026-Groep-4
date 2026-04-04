#!/bin/bash -l
# scripts/hpc/install.sh
#
set -eo pipefail

echo ">>> Starting HPC Installation in $(hostname)..."

# Mirror configs to $VSC_DATA to avoid home quota limits (3GB)
PROJ_NAME=$(basename "$PWD")
HPC_CONFIG_DIR="$VSC_DATA/$PROJ_NAME/env/hpc"
mkdir -p "$HPC_CONFIG_DIR"
cp env/hpc/*.txt "$HPC_CONFIG_DIR/"

module load vsc-venv

echo ">>> Synchronizing and activating environment (vsc-venv)..."
set +eo pipefail
source vsc-venv --activate \
    --modules "$HPC_CONFIG_DIR/modules.txt" \
    --requirements "$HPC_CONFIG_DIR/requirements.txt"
set -eo pipefail

# Overlay specific NumPy/Protobuf versions to ensure venv precedence
echo ">>> Applying library overlays (NumPy, Protobuf)..."
pip install --upgrade --no-deps numpy protobuf

echo '>>> Installing ipykernel...'
CLUSTER_ID="${VSC_INSTITUTE_CLUSTER:-generic}"
python -m ipykernel install --user --name="sel3_${CLUSTER_ID}" \
    --display-name "SEL3 (${CLUSTER_ID})"

echo '>>> Done'
