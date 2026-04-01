# Experiment Logger

A lightweight logging framework supporting Weights & Biases, local JSON, and stdout.

## Usage

```python
from experiment_logger import UnifiedLogger

logger = UnifiedLogger(run_name="my_experiment", config={"lr": 0.001})

logger.log({"loss": 0.5}, step=1)
logger.save_checkpoint(params=model_params, step=1)
logger.finish()
```

Logs and checkoints are saved in the `runs/` directory. If `track=True` (or `use_wandb=True`), everything is additionally synced to Weights & Biases.
