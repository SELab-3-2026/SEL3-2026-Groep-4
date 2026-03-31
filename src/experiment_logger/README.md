# Experiment Logger

A lightweight, standalone logging framework for machine learning experiments with multi-backend support.

## Features

- **Multi-backend logging**: Simultaneously log to WandB, local disk (JSON), and stdout
- **Data preservation**: All metrics saved locally, even if WandB is unavailable
- **Checkpoint management**: Save model checkpoints with metadata
- **WandB integration**: Optional artifact upload for model versioning
- **Graceful degradation**: Works without WandB installed
- **Simple API**: Minimal configuration required

## Installation

This package is included in the project. To use it in your code:

```python
from experiment_logger import UnifiedLogger
```

## Quick Start

```python
from experiment_logger import UnifiedLogger

# Initialize logger
logger = UnifiedLogger(
    run_name="my_experiment",
    config={"learning_rate": 0.001, "batch_size": 32},
    project_name="MyProject",
    entity="my-wandb-username",  # Optional
    use_wandb=True,  # Set to False to disable WandB
)

# Log metrics
for step in range(100):
    logger.log({
        "loss": 1.0 / (step + 1),
        "accuracy": step * 0.01,
    }, step=step)

# Save checkpoint
logger.save_checkpoint(
    params=model_params,
    step=100,
    metadata={"epoch": 1, "val_acc": 0.95},
)

# Save final model
logger.save_final_model(
    params=final_params,
    metadata={"final_accuracy": 0.98},
)

# Finalize (flushes remaining metrics)
logger.finish()
```

## Context Manager

Use as a context manager for automatic cleanup:

```python
with UnifiedLogger(run_name="my_exp", config={}) as logger:
    logger.log({"metric": 1.0})
    # Automatically calls finish() on exit
```

## Configuration

### Constructor Parameters

- `run_name` (str): Unique name for this run
- `config` (dict): Configuration dictionary with hyperparameters
- `project_name` (str): WandB project name (default: "PPO-Modularity")
- `entity` (str, optional): WandB entity (team/user name)
- `base_dir` (str): Base directory for local storage (default: "runs")
- `use_wandb` (bool): Enable WandB logging (default: True)
- `save_code` (bool): Save code to WandB (default: True)

### Directory Structure

```
runs/
└── my_experiment/
    ├── config.json          # Saved configuration
    ├── metrics/
    │   └── metrics.jsonl    # Line-delimited JSON metrics
    ├── checkpoints/
    │   ├── checkpoint_step_100.flax
    │   └── checkpoint_step_100_metadata.json
    └── final_model.flax
```

## API Reference

### `log(metrics, step=None, commit=True)`

Log metrics to all backends.

**Parameters:**
- `metrics` (dict): Dictionary of metric name -> value
- `step` (int, optional): Global step counter (auto-incremented if None)
- `commit` (bool): Whether to commit to WandB immediately

### `save_checkpoint(params, step, prefix="checkpoint", metadata=None)`

Save model checkpoint to disk and optionally to WandB.

**Parameters:**
- `params`: Model parameters (Flax params or any serializable object)
- `step` (int): Current training step
- `prefix` (str): Prefix for checkpoint filename
- `metadata` (dict, optional): Additional metadata to save

### `save_final_model(params, metadata=None)`

Save the final trained model.

**Parameters:**
- `params`: Model parameters
- `metadata` (dict, optional): Metadata about the final model

### `finish()`

Finalize logging and cleanup. Flushes remaining metrics to disk.

## Usage in Projects

This logger is designed to be:
- **Project-agnostic**: Use in any ML project, not just this one
- **Framework-agnostic**: Works with JAX, PyTorch, TensorFlow, etc.
- **Minimal dependencies**: Only requires `wandb` (optional), `flax` (for serialization), and `numpy`

## Design Philosophy

1. **Never lose data**: All metrics saved locally, regardless of WandB availability
2. **Simple API**: Minimal boilerplate, easy to integrate
3. **Fail gracefully**: Missing WandB shouldn't break experiments
4. **Reproducibility**: Save full configuration with every run

## License

Part of the 2026SEL3-project-BrittleStar repository.
