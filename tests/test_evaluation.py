import numpy as np
import pytest
import yaml
from pathlib import Path

from brittle_star_project.evaluation.checkpoint import (
    metadata_to_configs,
    TrainingConfig,
    load_metadata,
)
from brittle_star_project.evaluation.rollout import _maybe_clip_action
from brittle_star_project.environment.env_config import (
    MorphologyConfig,
    ArenaConfig,
    EnvConfig,
    ObservationBoundsConfig,
)
from brittle_star_project.environment.env_types import Task


def test_metadata_to_configs():
    """Test that a raw metadata dictionary correctly instantiates the typed configs."""
    mock_metadata = {
        "morphology": {
            "segments_per_arm": [4, 0, 4, 0, 0],
            "use_p_control": False,
        },
        "arena": {"sand_ground_color": False, "size": [15.0, 10.0]},
        "environment": {
            "task": "LIGHT_ESCAPE",
            "simulation_time": 5000.0,
        },
        "obs_bounds": {"joint_velocity": [-10.0, 10.0]},
    }

    config = metadata_to_configs(mock_metadata)

    assert isinstance(config, TrainingConfig)

    # Check MorphologyConfig
    assert isinstance(config.morphology, MorphologyConfig)
    assert config.morphology.segments_per_arm == [4, 0, 4, 0, 0]
    assert config.morphology.use_p_control is False
    assert config.morphology.use_torque_control is False  # default

    # Check ArenaConfig
    assert isinstance(config.arena, ArenaConfig)
    assert config.arena.sand_ground_color is False
    assert config.arena.size == [15.0, 10.0]
    assert config.arena.wall_height == 1.5  # default

    # Check EnvConfig
    assert isinstance(config.environment, EnvConfig)
    assert config.environment.task == Task.LIGHT_ESCAPE
    assert config.environment.simulation_time == 5000.0
    assert config.environment.time_scale == 2  # default

    # Check ObservationBoundsConfig
    assert isinstance(config.obs_bounds, ObservationBoundsConfig)
    assert config.obs_bounds.joint_velocity == [-10.0, 10.0]
    assert config.obs_bounds.segment_contact == [0.0, 1.0]  # default


def test_maybe_clip_action():
    """Test action clipping against boundaries."""
    # Test valid clipping
    action = np.array([1.5, -2.5, 0.0])
    low = np.array([-1.0, -1.0, -1.0])
    high = np.array([1.0, 1.0, 1.0])

    clipped = _maybe_clip_action(action, low, high)
    np.testing.assert_array_equal(clipped, np.array([1.0, -1.0, 0.0]))

    # Test skipping when bounds are None
    unclipped_1 = _maybe_clip_action(action, None, high)
    np.testing.assert_array_equal(unclipped_1, action)

    unclipped_2 = _maybe_clip_action(action, low, None)
    np.testing.assert_array_equal(unclipped_2, action)

    # Test skipping on shape mismatch
    wrong_low = np.array([-1.0, -1.0])  # Shape mismatch
    unclipped_3 = _maybe_clip_action(action, wrong_low, high)
    np.testing.assert_array_equal(unclipped_3, action)


def test_load_metadata_with_override(tmp_path: Path):
    """Test that metadata can be loaded from both default and override paths."""
    # 1. Setup
    model_path = tmp_path / "model.flax"
    model_path.write_bytes(b"dummy")

    default_metadata_path = tmp_path / "model_metadata.yaml"
    default_content = {"version": "default", "seed": 42}
    with open(default_metadata_path, "w") as f:
        yaml.dump(default_content, f)

    override_path = tmp_path / "custom_metadata.yaml"
    override_content = {"version": "override", "seed": 1337}
    with open(override_path, "w") as f:
        yaml.dump(override_content, f)

    # 2. Test default behavior
    loaded_default = load_metadata(model_path)
    assert loaded_default == default_content

    # 3. Test override behavior
    loaded_override = load_metadata(model_path, metadata_override_path=override_path)
    assert loaded_override == override_content

    # 4. Test Error Case
    non_existent = tmp_path / "missing.yaml"
    with pytest.raises(FileNotFoundError, match="Could not find metadata YAML at"):
        load_metadata(model_path, metadata_override_path=non_existent)
