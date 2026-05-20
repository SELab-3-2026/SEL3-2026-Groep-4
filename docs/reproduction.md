# Results & Reproduction

This guide explains how to access our official training logs and reproduce our results.

Our official training runs, model configurations, and metrics are publicly hosted on Weights & Biases (WandB).

---

## Weights & Biases (WandB) Project

All experiments, final models, and training logs are tracked in our public WandB project:

* **Official Runs Table**: [WandB final-models-v2 Table](https://wandb.ai/SEL3-2026-Groep-4/final-models-v2/table?nw=96mloffsyq)

This page lists the verified runs with their architecture types, morphology definitions, evaluation metrics, and final model performance.

### How to Reproduce a Run from WandB

Weights & Biases provides a built-in feature to extract the exact parameters and commands used for any given run:

1. Open the [WandB final-models-v2 Table](https://wandb.ai/SEL3-2026-Groep-4/final-models-v2/table?nw=96mloffsyq).
2. Click on the name of the run you wish to reproduce to open its detail page.
3. In the top-right corner of the run header (next to the run name, not the main workspace header), click the **three dots (`...`)** menu.
4. Select **"Reproduce run"**. This will display the exact command-line arguments and configuration settings used to execute that run.

---

## Local & HPC Reproduction Workflow

To reproduce our training and evaluation phases locally or on an HPC cluster, follow the procedures below.

### 1. Environment Setup

To ensure identical package versions (including JAX, Flax, and MuJoCo), sync your environment using the lockfile:

```bash
uv sync --frozen
```

### 2. Training Phase

Run the training script using the exact parameters retrieved from WandB's "Reproduce run" page or from a downloaded `_metadata.yaml` file:

```bash
uv run python scripts/train.py experiment=my_experiment ppo.learning_rate=0.001 experiment.seed=42
```

---

## Evaluation Phases

Reproducing our evaluation results is divided into two distinct phases:

### Phase 1: Determining the Best Checkpoint

During training, checkpoints are saved at regular intervals. To determine which of these checkpoints performed the best:

1. **Evaluate Checkpoints Post-Training**:
   If checkpoint evaluation was not run during training, scan the completed run's checkpoints folder by pointing to the final model path:

   ```bash
   uv run python scripts/evaluate_checkpoints.py simulation.model_path=runs/your_run_dir/final_model.flax
   ```

   This script runs deterministic rollouts for every checkpoint in `runs/your_run_dir/checkpoints/`.

2. **Locate the Results**:
   The evaluations are saved to:

   ```text
   runs/your_run_dir/metrics/checkpoint_evaluation.csv
   ```

   Analyze this CSV to find the checkpoint iteration with the highest average return or target success rate. This checkpoint will be used for cross-architecture comparisons.

### Phase 2: Comparing Checkpoints Between Architectures

Once the best checkpoints for each architecture are identified, they are compared under shared, standardized environments (including fault tolerance checks such as leg amputations).

1. **Configure the Comparison Models**:
   Open or create an evaluation config file (e.g., `configs/evaluation/poster.yaml`) and add the paths to the best checkpoints:

   ```yaml
   # configs/evaluation/poster.yaml
   evaluation:
     comparison_models:
       - runs/run_arch_centralized/checkpoints/checkpoint_best.flax
       - runs/run_arch_decentralized/checkpoints/checkpoint_best.flax
   ```

2. **Execute the Comparison Script**:
   Run the comparison script using your config:

   ```bash
   uv run python scripts/compare_models.py evaluation=poster
   ```

   This script runs multiple sequential evaluation episodes (defined by `comparison_num_episodes` starting at `comparison_base_seed`) for every model across the selected morphologies.

3. **Analyze Comparison Metrics**:
   The script writes a consolidated CSV file to `metrics/model_comparison.csv` containing:

   * **`eval_return`**: The cumulative return.
   * **`approx_max_velocity`**: The distance covered per step.
   * **`reached_target`**: Navigational success rates.
   * **`arm_0` to `arm_4`**: Active segments per arm (indicating damage/amputations).

This CSV can then be passed to the plotting scripts (e.g., `scripts/plots/analyze_comparisons.py`) to generate visualization plots. For details on configuration and outputs, see the **[Analysis & Plotting Guide](./api/analysis.md)**.
