# Brittle Star

## Usage

### UV

To set up the UV module, you can run the following command:

```bash
uv sync --frozen
```

example command:

```bash
uv run src/train.py --model_name my_model --epochs 50 --batch_size 32
```

## HPC Integration

To run training on the VSC HPC (Ghent Tier 2):

### Setup

Run the installation script once on a login node. This sets up a virtual environment in `$VSC_DATA` with the required system modules (JAX, Flax, WandB) and lightweight dependencies.

```bash
bash scripts/hpc_install.sh
```

### Smoke Test
Verify everything runs on a compute node:
```bash
python src/train.py --config configs/hpc_smoke_test.yaml
```

### Submitting Jobs
Submit long-running experiments via the provided Slurm script:
```bash
sbatch scripts/hpc_train.slurm
```

### Syncing WandB
Since compute nodes are offline, sync your WandB logs after the job finishes:
```bash
wandb sync runs/<run_name>/wandb
```
