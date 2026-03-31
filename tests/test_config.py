"""Tests for YAML config loading."""

import sys
from pathlib import Path

import pytest

# Ensure src is on the path when running from the project root
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

CONFIGS_DIR = Path(__file__).parent.parent / "configs"


class TestYamlConfig:
    def test_load_yaml_config(self):
        from experiment_logger.config_utils import load_yaml_config

        config = load_yaml_config(str(CONFIGS_DIR / "default_ppo.yaml"))
        assert isinstance(config, dict)
        assert "total_timesteps" in config
        assert "learning_rate" in config

    def test_load_dev_test_config(self):
        from experiment_logger.config_utils import load_yaml_config

        config = load_yaml_config(str(CONFIGS_DIR / "dev_test.yaml"))
        assert config["total_timesteps"] == 100000

    def test_missing_config_raises(self):
        from experiment_logger.config_utils import load_yaml_config

        with pytest.raises(FileNotFoundError):
            load_yaml_config("nonexistent.yaml")

    def test_merge_config_with_cli_is_callable(self):
        from experiment_logger.config_utils import merge_config_with_cli

        assert callable(merge_config_with_cli)
