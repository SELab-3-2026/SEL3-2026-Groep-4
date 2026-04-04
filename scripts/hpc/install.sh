#!/bin/bash
# scripts/hpc/install.sh
#
# Usage (from project root):
#   bash scripts/hpc/install.sh

set -eo pipefail

# Keep caches off $VSC_HOME (quota ~3 GB).
export PIP_CACHE_DIR="$VSC_SCRATCH/.cache/pip"
export UV_CACHE_DIR="$VSC_SCRATCH/.cache/uv"
mkdir -p "$PIP_CACHE_DIR" "$UV_CACHE_DIR"

module load vsc-venv

source vsc-venv --activate \
    --modules env/hpc/modules.txt \
    --requirements env/hpc/requirements.txt

python -m ipykernel install --user --name="sel3_${VSC_INSTITUTE_CLUSTER}" \
    --display-name "SEL3 (${VSC_INSTITUTE_CLUSTER})"
