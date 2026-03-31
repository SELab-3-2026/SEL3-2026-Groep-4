# Configuration Files

This directory contains configuration files for training experiments.

## Quick Start

### 1. Choose a Template

**For Development/Testing:**
```bash
cp configs/dev_test.yaml configs/my_dev.yaml
```

**For Production Training:**
```bash
cp configs/production_training.yaml configs/my_experiment.yaml
```

### 2. Configure Your Settings

Edit your config file and **set your wandb entity**:
```yaml
# ⚠️  IMPORTANT: Set this to your WandB username or team name
wandb_entity: "your-wandb-username"  
track: true  # Enable WandB logging
```

### 3. Run Training

**Using config file:**
```bash
python src/train.py --config configs/my_experiment.yaml
```

**Override specific parameters:**
```bash
python src/train.py --config configs/my_experiment.yaml --learning-rate 0.001 --num-envs 32
```

**Pure CLI (no config file):**
```bash
python src/train.py --track --wandb-entity your-username --total-timesteps 1000000
```

## Features

### 📊 WandB Integration  
- Real-time metrics logging
- Model checkpoints as artifacts
- Run comparison and collaboration

### 🔧 Flexible Configuration
- YAML files for reproducible experiments
- CLI overrides for quick adjustments  
- Team collaboration without code changes

## Configuration Templates

### `dev_test.yaml`
- Fast iteration for development
- Short runs (100K timesteps)
- Frequent checkpoints
- Small environment count

### `production_training.yaml`  
- Full-scale training (50M timesteps)
- Optimized hyperparameters
- Production-ready settings

### `default_ppo.yaml`
- Baseline configuration template
- Balanced settings for most use cases

## Team Collaboration

Each team member should create their own config file:

```yaml
# configs/alice_experiment.yaml
exp_name: "alice_locomotion_v2"
track: true
wandb_project_name: "PPO-Modularity"
wandb_entity: "alice-research"  # Alice's WandB username
total_timesteps: 20000000
# ... other settings
```

This allows everyone to:
- Use their own WandB account
- Run different experiments simultaneously  
- Share configurations via version control
- Avoid conflicts in run names
