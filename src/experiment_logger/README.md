# Experiment Logger

A standardized, unified interface for logging experiments across multiple backends (WandB, TensorBoard, and Local Disk).

This library is designed to be a standalone package that decouples the logging logic from the core training routines in the `brittle_star_project`.

## Quick Start

The recommended way to use the logger is through the `get_logger()` singleton:

```python
from experiment_logger import UnifiedLogger, get_logger

# Initialize at the start of your script (e.g., in train.py)
logger = UnifiedLogger(
    run_name="my_experiment_run",
    config={"learning_rate": 3e-4},
    project_name="MyProject",
    base_dir="runs",
    use_wandb=True
)

# In other files, retrieve the initialized singleton:
# logger = get_logger()

# Log metrics (Scalar values, numpy scalars, or JAX types)
logger.log({"loss": 0.5, "accuracy": 0.98}, step=100)

# Standard logging (Mirrored to disk and stdout)
logger.info("Training started")
logger.warning("Learning rate is very high")

# Save checkpoints (Automatically synced to WandB as artifacts)
logger.save_checkpoint(params, step=5000)
```

## Logger Classes

### `UnifiedLogger`

The full suite for production training. It manages:
- **WandB**: Syncs metrics and uploads model checkpoints as artifacts.
- **TensorBoard**: Writes events for local visualization.
- **Local Disk**: Stores metrics in `metrics.yaml` and textual logs in `run.log`.

### `SimpleLogger`

A zero-dependency fallback that uses standard Python `print()` statements. Use this for standalone testing or minimal environments where you don't need persistent monitoring.

```python
from experiment_logger import SimpleLogger
logger = SimpleLogger(run_name="test_run")
```

## API Features

### `logger.progress_bar(iterable, **kwargs)`

A smart wrapper around `tqdm` that automatically detects its environment. 
- **Interactive Terminal**: Displays a normal progress bar.
- **Non-Interactive (HPC)**: Automatically disables the bar to prevent log file bloat in `slurm.out`.

### `logger.log_non_interactive(msg: str)`

Prints a message *only* when running in non-interactive environments. Useful for high-level progress tracking (e.g., "Epoch 5 Complete") without interactive noise.

### `logger.save_checkpoint(params, step, prefix="checkpoint")`

Saves model parameters using Flax serialization.
- **Local Location**: `runs/<run_name>/checkpoints/`
- **WandB Logic**: Automatically uploads the `.flax` file as a model artifact for lineage tracking.
