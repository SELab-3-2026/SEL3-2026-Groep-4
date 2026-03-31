# Brittle Star

Reinforcement learning research on brittle star locomotion using PPO.

## Quick Start

### Installation

Set up the environment using UV:

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

Run training with your configuration:

```bash
uv run python src/train.py
```

Or use a custom config file:

```bash
uv run python src/train.py --config configs/my_experiment.yaml
```

Override specific parameters:

```bash
uv run python src/train.py --learning-rate 0.001 --num-envs 32 --track
```

### Logging

The training script uses a unified logging framework that:
- Logs to **WandB** (when enabled)
- Saves metrics to **local disk** (JSON files in `runs/`)
- Displays progress in **stdout**

All experiment data is preserved locally, even if WandB is unavailable.

## Project Structure

```
src/brittle_star_project/  # Core library (reusable components)
├── logging/               # Unified logging framework
├── environment/           # Environment wrappers
├── rl/                    # RL algorithms and models
└── dataclasses/          # Configuration dataclasses

configs/                   # Training configurations
runs/                      # Training outputs (checkpoints, metrics)
```

## For Researchers

**Important:** Do not commit your personal WandB credentials to the repository. 
Instead, create your own config file (e.g., `configs/yourname.yaml`) and add it to `.gitignore` if needed.

See [configs/README.md](configs/README.md) for more details on configuration management.
