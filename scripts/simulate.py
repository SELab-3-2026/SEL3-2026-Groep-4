"""Simulate a trained policy in the MuJoCo viewer.

Uses Hydra to load the same BrittleStarConfig that was used during training.
Override settings via CLI, e.g.:
    python scripts/simulate.py morphology=3_arms
"""

from __future__ import annotations

import hydra
from omegaconf import DictConfig, OmegaConf
from pathlib import Path

from brittle_star_project import (
    Backend,
    BrittleStarEnv,
    BrittleStarEnvFactory,
    SimulationConfig,
    simulate_policy,
)
from brittle_star_project.configs.main_config import BrittleStarConfig
from brittle_star_project.configs.register_configs import register_configs
from brittle_star_project.rl import RLModel
from brittle_star_project.rl.base import get_rl_model_registry

MODEL_BY_NAME = get_rl_model_registry()
MODEL_OPTIONS = sorted(MODEL_BY_NAME)


@hydra.main(config_path="../configs", config_name="main_config", version_base="1.3")
def main(dict_cfg: DictConfig) -> None:
    cfg: BrittleStarConfig = OmegaConf.to_object(dict_cfg)

    # Allow overriding these via Hydra CLI or a dedicated simulate config group
    # For now, defaults matching the old argparse behavior
    backend = Backend.MJX
    model_type = "random"
    model_path = None
    seed = cfg.experiment.seed

    # ======= ENVIRONMENT SETUP =======
    factory = BrittleStarEnvFactory()
    raw_env = factory.create_environment(backend, cfg.morphology, cfg.arena, cfg.environment)
    env = BrittleStarEnv(raw_env, backend=backend, config=cfg.environment)

    state = env.reset(seed=seed)

    # ======= MODEL SETUP =======
    nu = int(state.mj_model.nu)

    if model_path is not None:
        policy = RLModel.load(Path(model_path))
        if hasattr(policy, "nu"):
            policy.nu = nu
    else:
        model_cls = MODEL_BY_NAME[model_type]
        policy = model_cls(seed=seed)
        if hasattr(policy, "nu"):
            policy.nu = nu

    default_seed = int(getattr(policy, "seed", seed))

    # ======= SIMULATION =======
    rollout_cfg = SimulationConfig(
        realtime=True,
        seed=default_seed,
    )

    simulate_policy(policy, rollout_cfg, state)
    env.close()


if __name__ == "__main__":
    register_configs()
    main()
