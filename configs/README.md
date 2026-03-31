# Configuration Files

This directory contains configuration files for training experiments.

## Usage

Configuration files use YAML format and allow you to specify all training parameters in one place.

### Quick Start

Copy the default configuration template:
```bash
cp configs/default_ppo.yaml configs/my_experiment.yaml
```

Edit `my_experiment.yaml` to customize your experiment settings, particularly:
- `wandb_entity`: Your WandB username or team name
- `track`: Set to `true` to enable WandB logging
- Training hyperparameters as needed

Run training with your config:
```bash
python src/train.py --config configs/my_experiment.yaml
```

### Override Parameters

You can override any parameter from the command line:
```bash
python src/train.py --config configs/my_experiment.yaml --learning-rate 0.001 --num-envs 32
```

### Configuration for Different Users

Each researcher should create their own config file with their WandB settings:
```yaml
# configs/researcher_name.yaml
exp_name: "researcher_name_experiment"
track: true
wandb_project_name: "PPO-Modularity"
wandb_entity: "your-wandb-username"  # Change this!
```

This approach allows everyone to use the codebase without modifying source files.

## Available Configurations

- `default_ppo.yaml` - Default PPO training configuration template
