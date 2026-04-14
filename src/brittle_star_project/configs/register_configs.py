from hydra.core.config_store import ConfigStore

from experiment_logger.config_logger import LoggingConfig
from brittle_star_project.configs.config_experiment import ExperimentConfig
from brittle_star_project.configs.config_ppo import PPOConfig
from brittle_star_project.configs.config_networks import NetworksConfig, MLPConfig, LayerConfig
from brittle_star_project.environment.env_config import MorphologyConfig, ArenaConfig, EnvConfig
from brittle_star_project.configs.main_config import BrittleStarConfig


def register_configs():
    """Register dataclasses with Hydra's ConfigStore."""
    cs = ConfigStore.instance()

    # Store the main config schema
    cs.store(name="brittle_star_config", node=BrittleStarConfig)

    # Store individual structured configs for validation
    cs.store(group="experiment", name="base_experiment", node=ExperimentConfig)
    cs.store(group="logging", name="base_logging", node=LoggingConfig)
    cs.store(group="ppo", name="base_ppo", node=PPOConfig)
    cs.store(group="networks", name="base_networks", node=NetworksConfig)

    # Environment configs (reusing existing environment config objects)
    cs.store(group="morphology", name="base_morphology", node=MorphologyConfig)
    cs.store(group="arena", name="base_arena", node=ArenaConfig)
    cs.store(group="environment", name="base_environment", node=EnvConfig)
