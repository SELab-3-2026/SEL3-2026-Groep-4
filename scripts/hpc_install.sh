#!/bin/bash
# hpc_install.sh — Run once on a login node to set up the project environment.
# Usage: bash scripts/hpc_install.sh

set -euo pipefail

# -- Storage: put the venv on $VSC_DATA ---------------------------------------
mkdir -p "$VSC_DATA/venvs"
ln -sf "$VSC_DATA/venvs" "$VSC_HOME/venvs"

# -- Load built-in HPC modules to save space ----------------------------------
# Loading these will automatically load Python 3.11.3 + CUDA + GCC 12.3.0
ml load jax/0.4.25-gfbf-2023a-CUDA-12.1.1
ml load Flax/0.8.4-gfbf-2023a-CUDA-12.1.1
ml load Optax/0.2.2-gfbf-2023a-CUDA-12.1.1
ml load wandb/0.16.1-GCC-12.3.0
ml load matplotlib/3.7.2-gfbf-2023a
ml load PyYAML/6.0-GCCcore-12.3.0
ml load FFmpeg/5.1.2-GCCcore-12.3.0

export VENV_PATH="$VSC_DATA/venvs/sel3_${VSC_INSTITUTE_CLUSTER}"

# -- Create venv with system-site-packages so it sees the loaded modules ----
python -m venv --system-site-packages "$VENV_PATH"
source "$VENV_PATH/bin/activate"

# -- Install only the missing lightweight packages ---------------------------
# We bypass uv here to strictly adhere to the user's storage limits requirement.
pip install --upgrade pip
pip install biorobot==0.4.2 evosax==0.2.0 mediapy==1.2.6 tyro>=1.0.10 \
    cleanrl>=0.4.8 gymnasium>=1.2.3 PyOpenGL>=3.1.10 PyOpenGL-accelerate>=3.1.10

# -- Register as Jupyter kernel -----------------------------------------------
python -m ipykernel install --user --name="sel3_${VSC_INSTITUTE_CLUSTER}-kernel"

echo "✅ Environment created at $VENV_PATH"
