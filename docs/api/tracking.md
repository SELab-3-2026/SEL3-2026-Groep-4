# Tracking & Monitoring

This guide explains how to monitor your experiments using Weights & Biases (WandB) and TensorBoard.

## Weights & Biases (WandB)

WandB is used for online synchronization and visualization of training metrics.

### Authorization

Export your API key in your terminal to enable WandB synchronization:

```bash
export WANDB_API_KEY=your_copied_api_key_here
```

Alternatively, you can log in using the CLI:

```bash
uv run wandb login
```

### Enabling Tracking

To enable online sync during a training run, set `logging.track=true` on the command line:

```bash
uv run python scripts/train.py logging.track=true
```

You can also configure your project and entity:

```bash
uv run python scripts/train.py \
    logging.track=true \
    logging.wandb_project_name="MyProject" \
    logging.wandb_entity="my-team"
```

These can also be set in your configuration YAML file under the `logging` key.

## Local Monitoring with TensorBoard

All runs are recorded locally in the `runs/` directory (or the directory specified in `experiment.base_run_dir`). You can view scalars and other metrics with TensorBoard:

```bash
tensorboard --logdir runs/
```

Access the interface at `http://localhost:6006`.

### CLI Exploration Tool

For quick diagnostics or to export data to CSV without launching the full TensorBoard UI, you can use the `explore_tensorboard.py` script:

```bash
uv run python scripts/analysis/explore_tensorboard.py runs/your_run_name/
```

See the detailed description in [`/scripts/analysis/README.md`](../../scripts/analysis/README.md).
