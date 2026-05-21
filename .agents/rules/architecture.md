# Architecture and Code Formatting Rules

When writing or modifying code in this project, adhere strictly to the following rules:

## 1. Core Frameworks & Tooling
- **PPO & CleanRL**: Proximal Policy Optimization (PPO) is the baseline algorithm. Use CleanRL as the starting framework for PPO, ensuring adaptation for continuous action spaces.
- **JAX / Flax**: All Artificial Neural Network (ANN) controller architectures must be implemented using Flax (neural networks in JAX). Ensure full compatibility with the JAX/Flax ecosystem.
- **MuJoCo**: The simulation environment uses a MuJoCo brittle star. Ensure that XML structures (sensors, actuators, joints, morphology) respect realistic constraints and adhere to the project requirements.

## 2. Clean Code Principles
- **Naming Conventions**: Variables and functions must have consistent, intention-revealing names. A descriptive name is universally preferred over a short name with an explanatory comment.
- **Single Responsibility Function Design**: Functions must be modular. Minimize arguments and completely avoid boolean flag arguments that control execution behavior.
- **Commenting**: Code explains the "how". Comments are strictly reserved for explaining the "why". 
- **No Commented-out Code**: The AI must **never** generate commented-out or dead code. Delete it using version control instead.
- **YAGNI & Complexity Management**: You Aren't Gonna Need It. Avoid premature optimization or unnecessary abstraction. Only introduce complexity with documented justification. Break large functions into small, testable blocks.
- **No Notebooks for Core Logic**: Jupyter Notebooks are explicitly forbidden for general software development as they discourage modularity. They should only be used for prototyping, tutorials, or post-processing analysis.

## 3. Formatting & Linting
- **Ruff**: Output perfectly formatted code adhering to the Google style standard. Always format and lint the code using `ruff` (see `pyproject.toml` and `ruff.toml`).
- **Separation of Concerns**: Configuration code must be completely separated from implementation logic. Core logic must never be mixed with scripts or notebooks.
