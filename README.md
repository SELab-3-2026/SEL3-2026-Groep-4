# Brittle Star

## Quick Start

### Installation

To set up the UV module, you can run the following command:

```bash
uv sync --frozen
```

### Configuration

1. **Copy the default configuration:**
   ```bash
   cp configs/default_ppo.yaml configs/my_experiment.yaml
   ```

2. **Edit `configs/my_experiment.yaml`** to set your WandB credentials:
   ```yaml
   track: true  # Enable WandB logging
   wandb_entity: "your-wandb-username"  # Replace with your username/team
   wandb_project_name: "PPO-Modularity"
   ```

3. **(Optional) Login to WandB:**
   ```bash
   uv run wandb login
   ```

### Training

example command:

```bash
uv run python scripts/train.py
```

Or use a custom config file:

```bash
uv run python scripts/train.py --config configs/my_experiment.yaml
```

Override specific parameters:

```bash
uv run python scripts/train.py --learning-rate 0.001 --num-envs 32 --track
```

### Logging

The training script uses a unified logging framework that:
- Logs to **WandB** (when enabled)
- Saves metrics to **local disk** (JSON files in `runs/`)
- Displays progress in **stdout**

All experiment data is preserved locally, even if WandB is unavailable.

## HPC

See **[docs/HPC.md](docs/HPC.md)** for the full guide, including environment setup, cluster selection, interactive debugging, and job submission.
