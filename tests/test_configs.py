from hydra import compose, initialize
from omegaconf import OmegaConf
from brittle_star_project.configs.main_config import BrittleStarConfig
from brittle_star_project.configs.register_configs import register_configs

# Registration must happen before composition to enable validation against schemas
register_configs()


def test_config_composition_centralized():
    """Test that the centralized configuration composes and validates correctly."""
    with initialize(version_base="1.3", config_path="../configs"):
        # We compose the config; it follows main_config.yaml
        cfg = compose(config_name="main_config", overrides=["architecture=centralized"])

        # Merge with the dataclass class to get a structured DictConfig,
        # then convert to a real dataclass instance to verify validation.
        structured_cfg = OmegaConf.to_object(OmegaConf.merge(BrittleStarConfig, cfg))

        # Basic assertions
        assert "CentralizedConfig" in str(type(structured_cfg.architecture))
        assert isinstance(structured_cfg.ppo.learning_rate, float)
        assert structured_cfg.ppo.learning_rate > 0


def test_config_composition_decentralized():
    """Test that the decentralized configuration composes and validates correctly."""
    with initialize(version_base="1.3", config_path="../configs"):
        cfg = compose(config_name="main_config", overrides=["architecture=decentralized"])

        # Merge and convert to dataclass instance
        structured_cfg = OmegaConf.to_object(OmegaConf.merge(BrittleStarConfig, cfg))

        # Basic assertions
        assert "DecentralizedConfig" in str(type(structured_cfg.architecture))
        assert isinstance(structured_cfg.ppo.learning_rate, float)
        assert structured_cfg.ppo.learning_rate > 0

        # Decentralized specifics
        assert hasattr(structured_cfg.architecture, "message_passing_steps")
        assert structured_cfg.architecture.message_passing_steps > 0
