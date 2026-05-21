# Brittle Star

> What is the impact of different levels of controller-modularity on the learning-speed, coordination and tolerance for
defects (e.g. amputations) in brittle-star-like robots trained with Reinforcement Learning?

## Quick start

### Local setup

To set up the UV module, you can run the following command:

```bash
uv sync --frozen
```

## Repository Structure

```text
.
├── configs/                # Hydra configuration files (YAML)
├── docs/                   # Comprehensive documentation and API guides
├── runs/                   # Default output directory for Hydra and training artifacts
├── scripts/                # High-level entrypoints for training, simulation, and evaluation
├── src/
│   ├── brittle_star_project/ # Core library and environment logic
│   │   ├── evaluation/     # Checkpoint evaluation, rollout logic, and metrics persistence
│   │   └── trainers/       # Training loop implementations (e.g., PPO)
│   └── experiment_logger/  # Standalone logging package
└── tests/                  # Unit and integration tests
```

## Usage

For detailed instructions on how to use the project, please refer to the **[API Documentation](docs/README.md)**.

### Quick Start

1. **Train a model:**
   ```bash
   uv run python scripts/train.py ppo.learning_rate=0.001 logging.track=true
   ```

2. **Monitor progress:**
   See [Tracking & Monitoring](docs/api/tracking.md).

3. **Simulate a trained model:**
   See [Simulation & Evaluation](docs/api/simulation.md).

4. **Compare fault tolerance of models:**
   See [Checkpoint & Model Evaluation](docs/api/evaluation.md)

## Results & Reproduction

See **[docs/api/reproduction.md](docs/api/reproduction.md)** to learn how to access our public [Weights & Biases (WandB) project](https://wandb.ai/SEL3-2026-Groep-4/final-models-v2?nw=96mloffsyq), retrieve specific run parameters, and run the training/evaluation reproduction workflow.

## HPC

See **[docs/HPC.md](docs/HPC.md)** for the full guide, including environment setup, cluster selection, interactive debugging, and job submission.

## Documentation

Please find all documentation and a starting point for more information in [corresponding README](./docs/README.md).
