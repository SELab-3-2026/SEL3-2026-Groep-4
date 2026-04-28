# Brittle Star

> What is the impact of different levels of controller-modularity on the learning-speed, coordination and tolerance for
defects (e.g. amputations) in brittle-star-like robots trained with Reinforcement Learning?

## Quick start

### Local setup

To set up the UV module, you can run the following command:

```bash
uv sync --frozen
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

## HPC

See **[docs/HPC.md](docs/HPC.md)** for the full guide, including environment setup, cluster selection, interactive debugging, and job submission.

## Documentation

Please find all documentation and a starting point for more information in [corresponding README](./docs/README.md).
