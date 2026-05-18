# Checkpoint & Model Evaluation

This guide covers how to evaluate trained brittle star models, with a focus on measuring defect tolerance (amputations) across different controller architectures.

## Checkpoint Evaluation (During Training)

The `PPOTrainer` can automatically evaluate every saved checkpoint using the fast MJX backend. This is enabled via configuration.

### Configuration

In your experiment config or via CLI:
```bash
python scripts/train.py evaluation.evaluate_checkpoints=true evaluation.eval_max_steps=5000
```

Results are saved to `runs/<run_dir>/metrics/checkpoint_evaluation.csv` and synced to Weights & Biases if enabled.

## Cross-Model & Fault Tolerance Analysis

To measure how well different controllers handle damage (amputations), use `scripts/compare_models.py`. This script performs a grid search over models x morphologies.

1. Create or update a YAML file in `configs/evaluation`.
2. Run the benchmark:

```bash
python scripts/compare_models.py evaluation=poster
```

The script will evaluate every combination of model and morphology for the specified number of episodes.

The results are saved to a CSV (default: `metrics/model_comparison.csv`).

### CSV Schema

| Column                | Description                                                  |
|-----------------------|--------------------------------------------------------------|
| `model_path`          | Path to the trained weights.                                 |
| `architecture`        | The `morph_mode` of the model (e.g., `CENTRALIZED`, `RING`). |
| `arm_0` ... `arm_4`   | Number of segments in each arm slot (0 = amputated).         |
| `num_active_arms`     | Total number of arms with segments > 0.                      |
| `seed`                | The episode seed.                                            |
| `eval_return`         | Accumulated shaped reward.                                   |
| `approx_max_velocity` | Average velocity: `(initial_dist - final_dist) / steps`.     |
| `reached_target`      | Whether the robot finished within the success radius.        |

## Post-hoc Checkpoint Scanning

If you need to re-evaluate every saved checkpoint in a run (e.g., to generate a learning curve with different metrics):

```bash
python scripts/evaluate_checkpoints.py \
    simulation.model_path=runs/<run_id>/final_model.flax \
    evaluation.eval_max_steps=2000
```

This script scans the `checkpoints/` directory of the specified run and evaluates every `.flax` file it finds using the model's training morphology.
