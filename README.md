# Brittle Star

> What is the impact of different levels of controller-modularity on the learning-speed, coordination and tolerance for
defects (e.g. amputations) in brittle-star-like robots trained with Reinforcement Learning?

## Usage

### Local setup

To set up the UV module, you can run the following command:

```bash
uv sync --frozen
```

example command:

```bash
uv run src/train.py --model_name my_model --epochs 50 --batch_size 32
```

### HPC setup

See **[docs/HPC.md](docs/HPC.md)** for the full guide, including environment setup, cluster selection, interactive debugging, and job submission.

## Documentation

Please find all documentation and a starting point for more information in [corresponding README](./docs/README.md).
