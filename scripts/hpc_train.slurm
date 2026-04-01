#!/bin/bash
#SBATCH --job-name=brittle-star-ppo
#SBATCH --output=runs/slurm_%j.out
#SBATCH --error=runs/slurm_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --gpus=1
# Adjust --partition to your cluster (check with `sinfo` on the login node)
#SBATCH --partition=gpu

# -- Load the exact same modules as in hpc_install.sh -------------------------
ml load jax/0.4.25-gfbf-2023a-CUDA-12.1.1
ml load Flax/0.8.4-gfbf-2023a-CUDA-12.1.1
ml load Optax/0.2.2-gfbf-2023a-CUDA-12.1.1
ml load wandb/0.16.1-GCC-12.3.0
ml load matplotlib/3.7.2-gfbf-2023a
ml load PyYAML/6.0-GCCcore-12.3.0
ml load FFmpeg/5.1.2-GCCcore-12.3.0

# -- Activate the system-site-packages venv -----------------------------------
source "$VSC_DATA/venvs/sel3_${VSC_INSTITUTE_CLUSTER}/bin/activate"

# -- HPC-specific environment flags -------------------------------------------
export MUJOCO_GL=egl          # Headless OpenGL via EGL (no display required)
export WANDB_MODE=offline     # Sync to WandB after the job (no outbound internet needed)

# -- Run from the project root -------------------------------------------------
cd "$SLURM_SUBMIT_DIR"

python src/train.py --config configs/production_training.yaml
