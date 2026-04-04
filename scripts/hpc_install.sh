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

# -- Extract and install missing dependencies ----------------------------------
# Dynamically parse pyproject.toml and install only what is missing natively
MISSING_DEPS=$(python -c '
import tomllib, importlib.util, re
with open("pyproject.toml", "rb") as f:
    deps = tomllib.load(f)["project"]["dependencies"]

missing = []
for dep in deps:
    pkg = re.split(r"[\[=><~]", dep)[0].strip()
    # Map common PyPI package names to their python import names
    mapping = {"pyyaml": "yaml", "pyopengl": "OpenGL", "pyopengl-accelerate": "OpenGL_accelerate"}
    import_name = mapping.get(pkg.lower(), pkg.replace("-", "_"))
    
    # If the system module cannot find the package, add it to our pip install list
    if getattr(importlib.util, "find_spec", None) is None or importlib.util.find_spec(import_name) is None:
        missing.append(f"\"{dep}\"")

print(" ".join(missing))
')

if [ -n "$MISSING_DEPS" ]; then
    echo "Installing missing dependencies: $MISSING_DEPS"
    pip install --upgrade pip
    eval "pip install $MISSING_DEPS"
else
    echo "All dependencies from pyproject.toml are successfully satisfied by HPC system modules."
fi

# -- Register as Jupyter kernel -----------------------------------------------
python -m ipykernel install --user --name="sel3_${VSC_INSTITUTE_CLUSTER}-kernel"

echo "✅ Environment created at $VENV_PATH"
