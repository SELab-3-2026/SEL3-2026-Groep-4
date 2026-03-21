# Methodology and Process Rules

The agent must adhere to the following scientific and operational practices, focused on robustness and reproducibility:

## 1. Scientific Context & Methodology
- **Research Focus**: Maintain focus on the project's objective: studying how controller modularity affects learning speed, coordination, and fault tolerance in brittle-star locomotion.
- **Hypothesis-Driven Design**: Base execution on clear hypotheses. Document all design decisions prior to implementation (in `/docs/decisions/` or via Artifacts/Plans).
- **Scaffolding Approach**: Start development with simple setups before scaling to complex environments and varying morphologies.
- **Value of Negative Results**: Understand that a controller failing to learn locomotion, when coupled with a thorough analysis of the failure, holds strong scientific value. Do not artificially force a positive result.

## 2. Reproducibility Protection
- **Dependency Management (uv)**: This project strictly prefers `uv`. Do **not** manually modify the `uv.lock` file. Add dependencies via `uv add <package>` and sync environments via `uv sync --frozen` (or via devcontainers).
- **Consistent Initialization**: The AI must avoid hidden randomness. Ensure that all runs use consistent seed initialization.
- **Experiment Tracking**: Use Weights & Biases (wandb) for tracking and logging all experiments and parameters. Ensure every experiment-friendly parameter is properly logged.
