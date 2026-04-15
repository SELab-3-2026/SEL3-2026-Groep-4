from pathlib import Path
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf
from brittle_star_project.configs.main_config import BrittleStarConfig
from brittle_star_project.configs.register_configs import register_configs

# Registration must happen before composition to enable validation against schemas
register_configs()


def test_config_composition_centralized():
    """Test that the centralized configuration composes and validates correctly."""
    config_dir = str(Path(__file__).parent.parent / "configs")
    with initialize_config_dir(version_base="1.3", config_dir=config_dir):
        # We compose the config; it follows main_config.yaml
        cfg = compose(config_name="main_config", overrides=["architecture=centralized"])

        # Merge with the structured schema and convert to a real dataclass instance
        structured_cfg = OmegaConf.to_object(
            OmegaConf.merge(OmegaConf.structured(BrittleStarConfig), cfg)
        )

        # Basic assertions
        assert structured_cfg.architecture.name == "centralized"
        assert structured_cfg.architecture.propagator is None
        assert isinstance(structured_cfg.ppo.learning_rate, float)
        assert structured_cfg.ppo.learning_rate > 0


def test_config_composition_decentralized():
    """Test that the decentralized configuration composes and validates correctly."""
    config_dir = str(Path(__file__).parent.parent / "configs")
    with initialize_config_dir(version_base="1.3", config_dir=config_dir):
        cfg = compose(config_name="main_config", overrides=["architecture=decentralized"])

        # Merge and convert to dataclass instance
        structured_cfg = OmegaConf.to_object(
            OmegaConf.merge(OmegaConf.structured(BrittleStarConfig), cfg)
        )

        # Basic assertions
        assert structured_cfg.architecture.name == "decentralized"
        assert isinstance(structured_cfg.ppo.learning_rate, float)
        assert structured_cfg.ppo.learning_rate > 0

        # Decentralized specifics
        assert hasattr(structured_cfg.architecture, "message_passing_steps")
        assert structured_cfg.architecture.message_passing_steps > 0
