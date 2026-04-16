from hydra.core.config_store import ConfigStore

from experiment_logger.config_logger import LoggingConfig
from brittle_star_project.configs.config_experiment import ExperimentConfig
from brittle_star_project.configs.config_ppo import PPOConfig
from brittle_star_project.configs.config_architecture import (
    CentralizedConfig,
    DecentralizedConfig,
)
from brittle_star_project.configs.config_simulation import SimulationSettings
from brittle_star_project.environment.env_config import MorphologyConfig, ArenaConfig, EnvConfig
from brittle_star_project.configs.main_config import BrittleStarConfig


def register_configs() -> None:
    """Register all dataclasses with Hydra's ConfigStore.

    This must be called before hydra.main() processes the config, ensuring
    every structured config is validated against its Python schema. Typos in
    YAML keys will raise ConfigAttributeError at startup.
    """
    cs = ConfigStore.instance()

    # Root schema
    cs.store(name="brittle_star_config", node=BrittleStarConfig)

    # Sub-config groups — each group corresponds to a configs/ subdirectory.
    cs.store(group="experiment", name="base_experiment", node=ExperimentConfig)
    cs.store(group="logging", name="base_logging", node=LoggingConfig)
    cs.store(group="ppo", name="base_ppo", node=PPOConfig)

    # Architecture variants — swap via CLI: architecture=decentralized
    cs.store(group="architecture", name="centralized_schema", node=CentralizedConfig)
    cs.store(group="architecture", name="decentralized_schema", node=DecentralizedConfig)

    # Environment configs
    cs.store(group="morphology", name="base_morphology", node=MorphologyConfig)
    cs.store(group="arena", name="base_arena", node=ArenaConfig)
    cs.store(group="environment", name="base_environment", node=EnvConfig)
    cs.store(group="simulation", name="base_simulation", node=SimulationSettings)
