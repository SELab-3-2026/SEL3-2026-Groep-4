# Checkpoint & Model Evaluation

This guide covers how to evaluate trained brittle star models, both during training and as a post-training analysis step.

## Checkpoint Evaluation (During Training)

The `PPOTrainer` can automatically evaluate every saved checkpoint using the fast MJX backend. This is enabled via configuration.

### Configuration

In your experiment config or via CLI:
```bash
python scripts/train.py evaluation.evaluate_checkpoints=true evaluation.eval_max_steps=5000
```

Results are saved to `runs/<run_dir>/metrics/checkpoint_evaluation.csv` and synced to Weights & Biases if enabled.

## Cross-Model Comparison

To compare different architectures or runs, use `scripts/compare_models.py`.

1. Create or update a YAML file in `configs/evaluation/`.
2. Run the Comparison:

```bash
python scripts/compare_models.py evaluation=poster
```

The script will run the specified number of episodes for each model and produce a single CSV with return and velocity metrics.

## Post-hoc Checkpoint Evaluation

If you didn't enable evaluation during training, or want to re-run it with different settings, use `scripts/evaluate_checkpoints.py`.

```bash
python scripts/evaluate_checkpoints.py \
    simulation.model_path=runs/<run_id>/final_model.flax \
    evaluation.eval_seed=42
```

This script scans the `checkpoints/` directory and evaluates every `.flax` file it finds.

## Metrics Explained

- **`eval_return`**: The accumulated shaped reward using the `reward_fn` defined in `PPOTrainer`.
- **`approx_max_velocity`**: Calculated as `(initial_dist - final_dist) / total_steps`. Note that this is an average velocity over the episode.
- **`reached_target`**: Boolean indicating if the robot reached the target within the max steps.
