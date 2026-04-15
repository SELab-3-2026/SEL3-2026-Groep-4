# Configuration Files

This directory contains configuration files for training experiments.

## Usage

Use `--config` with `scripts/train.py` to run an experiment:

```bash
python scripts/train.py --config configs/default_ppo.yaml
```

You can overriding settings via CLI:
```bash
python scripts/train.py --config configs/default_ppo.yaml --learning-rate 0.001
```

## Available Configurations

- `default_ppo.yaml`: Baseline config.
- `dev_test.yaml`: Fast iteration for development.
- `production_training.yaml`: Full-scale training.
- `personal_template.yaml`: Template for team members to customize.
