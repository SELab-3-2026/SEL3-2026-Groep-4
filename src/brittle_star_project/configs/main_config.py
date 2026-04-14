from dataclasses import dataclass, field

from experiment_logger.config_logger import LoggingConfig
from brittle_star_project.configs.config_experiment import ExperimentConfig
from brittle_star_project.configs.config_ppo import PPOConfig
from brittle_star_project.configs.config_networks import NetworksConfig
from brittle_star_project.environment.env_config import MorphologyConfig, ArenaConfig, EnvConfig


@dataclass
class BrittleStarConfig:
    experiment: ExperimentConfig = field(default_factory=ExperimentConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    ppo: PPOConfig = field(default_factory=PPOConfig)
    networks: NetworksConfig = field(default_factory=NetworksConfig)
    morphology: MorphologyConfig = field(default_factory=MorphologyConfig)
    arena: ArenaConfig = field(default_factory=ArenaConfig)
    environment: EnvConfig = field(default_factory=EnvConfig)
