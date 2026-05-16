# Training Models

This guide covers how to configure and run training experiments for the Brittle Star project using Hydra-based configurations.

## Configuration

The project uses a modular configuration system powered by [Hydra](https://hydra.cc/). Instead of passing many command-line flags, you select and override configuration groups.

### Creating a Custom Experiment

1. **Create a new experiment file:**
   Create a file at `configs/experiment/my_experiment.yaml`. You can copy an existing one as a template:
   ```bash
   cp configs/experiment/base.yaml configs/experiment/my_experiment.yaml
   ```

2. **Edit `configs/experiment/my_experiment.yaml`** to set your experiment parameters:
   ```yaml
   # @package _global_
   experiment:
     exp_name: "my_custom_run"
     seed: 42
   ```

## Training Execution

To start a training run with the default settings defined in `configs/main_config.yaml`:

```bash
uv run python scripts/train.py
```

### Using a Custom Experiment Configuration

To run with your custom experiment file:

```bash
uv run python scripts/train.py experiment=my_experiment
```

```bash
uv run python scripts/train.py ppo.learning_rate=0.001 ppo.num_envs=32 logging.track=true
```

## Evaluation During Training

By default, the trainer saves checkpoints but does not evaluate them. To enable automatic headless evaluation of every saved checkpoint, set `evaluation.evaluate_checkpoints=true`:

```bash
uv run python scripts/train.py evaluation.evaluate_checkpoints=true
```

For more details on evaluation metrics and comparison tools, see [Evaluation](./evaluation.md).

For more details on tracking your experiments, see [Tracking & Monitoring](./tracking.md).
