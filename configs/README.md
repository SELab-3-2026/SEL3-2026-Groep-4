# Brittle Star Configuration System

This project uses **Hydra** for a modular, hierarchical, and strictly-typed configuration system.

## Core Concepts

1.  **Composition over Inheritance**: Instead of one giant config file, the configuration is composed of small, domain-specific modules (PPO settings, architecture, morphology, etc.).
2.  **Strict Typing**: Every configuration is validated against a Python dataclass schema (`ConfigStore`). Misspelled keys throw a `ConfigAttributeError` immediately.
3.  **CLI Swapping**: You can swap entire modules or override individual values from the command line without touching code.

## Directory Structure

- `main_config.yaml`: The root entry point defining the default composition.
- `experiment/`: High-level experiment settings (seed, device).
- `logging/`: WandB and checkpointing configuration.
- `ppo/`: PPO training hyperparameters.
- `architecture/`: Polymorphic network architectures (centralized vs. decentralized).
- `morphology/`: Physical robot definitions (number of segments, amputations).
- `arena/`: Environment physics and visual settings.
- `environment/`: Task-specific settings (Directed Locomotion, Light Escape).

## Common Commands

### Local Debugging
Run a quick test with minimal iterations:
```bash
python scripts/train.py experiment=dev_test ppo=fast
```

### Swapping Architectures or Morphologies
Test a decentralized controller on a 3-arm robot:
```bash
python scripts/train.py architecture=decentralized morphology=3_arms
```

### HPC Production
Run stable PPO with WandB enabled (HPC submission scripts handle the `hydra.run.dir` redirection):
```bash
python scripts/train.py ppo=stable logging=wandb_enabled
```

### Dry-Run Validation
Check if your configuration is valid without starting the simulation:
```bash
python scripts/train.py --cfg job
```

## Developer Notes

- **Adding a new group**: Create a subdirectory in `configs/` and register the new dataclass in `src/brittle_star_project/configs/register_configs.py`.
- **Typo Catching**: If you see a `ConfigAttributeError`, check for typos in your YAML keys or CLI overrides.
- **Output Redirection**: Hydra automatically creates `outputs/` directories. On HPC, ensure `hydra.run.dir` is set to a fast scratch storage.
