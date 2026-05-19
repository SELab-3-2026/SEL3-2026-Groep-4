# Documentation

Welcome to the Brittle Star project documentation. This codebase contains the implementations and research for the scientific evaluation of controller modularity in brittle-star-like robots trained using Reinforcement Learning.

For the core codebase, scripts, and contribution history, visit our [GitHub Repository](https://github.com/SELab-3-2026/SEL3-2026-Groep-4).

## Design & architecture (`/design`)

If you are interested in the "why did you do it like this?"

- [Actor/critic architecture](./design/actor-critic.md): Description of the actor-critic pipeline.
- [Communication](./design/communication.md): Message propagation, Nerve-Net style.
- [Controllers](./design/controllers.md): Macroscopig brain toplogy, centralized, arm-level, segment-level.
- [Input/output](./design/input_action_spaces.md): Description of the model's input and output.
- [Learning algorithm](./design/learning_algorithm.md): RL techniques, i.e. PPO.
- [Reward function](./design/reward_function.md): Goals, fitness tracking, and reward structures.

## API reference (`/api`)

If you are interested in the "how do I use it?"

- [Training](./api/training.md): How to configure and run experiments.
- [Tracking & Monitoring](./api/tracking.md): Setting up WandB and TensorBoard to monitor runs.
- [Simulation](./api/simulation.md): Visualizing and evaluating models.
- [Environment](./api/environment.md): MuJoCo environment interaction and configuration.
- [Analysis](./api/analysis.md): Comparing checkpoints and generating plots.
- [Evaluation](./api/evaluation.md): Evaluating checkpoints and comparing fault tolerance.