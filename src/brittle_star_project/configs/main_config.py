from dataclasses import dataclass, field

from experiment_logger.config_logger import LoggingConfig
from brittle_star_project.configs.config_experiment import ExperimentConfig
from brittle_star_project.configs.config_ppo import PPOConfig
from brittle_star_project.configs.config_architecture import ArchitectureConfig
from brittle_star_project.environment.env_config import MorphologyConfig, ArenaConfig, EnvConfig


@dataclass
class BrittleStarConfig:
    """Root configuration for a brittle star training run.

    Composed of strictly separated sub-configs. Each sub-config can be swapped
    independently via CLI or a different YAML file. See configs/README.md.
    """

    experiment: ExperimentConfig = field(default_factory=ExperimentConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    ppo: PPOConfig = field(default_factory=PPOConfig)
    # This field is polymorphic; defaults to the base class to allow subclasses
    # (CentralizedConfig, DecentralizedConfig) to be merged in via Hydra.
    architecture: ArchitectureConfig = field(default_factory=ArchitectureConfig)
    morphology: MorphologyConfig = field(default_factory=MorphologyConfig)
    arena: ArenaConfig = field(default_factory=ArenaConfig)
    environment: EnvConfig = field(default_factory=EnvConfig)
