# Documentation

## Design & architecture (`/design`)

If you are interested in the "why did you do it like this?"

- [Actor/critic architecture](./design/actor-critic.md): Description of the actor-critic pipeline.
- [Communication](./design/communication.md): Message propagation, Nerve-Net style.
- [Controllers](./design/controllers.md): Macroscopig brain toplogy, centralized, arm-level, segment-level.
- [Input/output](./design/input_action_spaces.md): Description of the model's input and output.
- [Learning algorithm](./design/learning_algorithm.md): RL techniques, i.e. PPO.
- [Reward function](./design/learning_algorithm.md): Goals, fitness tracking, and reward structures.

## API reference (`/api`)

If you are interested in the "how do I use it?"

- [Training](./api/training.md): How to configure and run experiments.
- [Tracking & Monitoring](./api/tracking.md): Setting up WandB and TensorBoard to monitor runs.
- [Simulation](./api/simulation.md): Visualizing and evaluating models.
- [Environment](./api/environment.md): MuJoCo environment interaction and configuration.
