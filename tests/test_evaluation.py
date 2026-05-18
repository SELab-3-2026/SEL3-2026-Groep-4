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
    MorphMode,
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


@pytest.fixture
def mock_training_config():
    return TrainingConfig(
        morphology=MorphologyConfig(
            segments_per_arm=[1, 1, 1, 1, 1], morph_mode=MorphMode.CENTRALIZED
        ),
        arena=ArenaConfig(),
        environment=EnvConfig(),
        obs_bounds=ObservationBoundsConfig(),
    )


@pytest.fixture
def mock_metadata():
    return {"architecture": {"message_passing_steps": 2}}


def test_build_eval_env_training_morphology(tmp_path, mock_training_config, mock_metadata):
    from brittle_star_project.evaluation.eval_env_builder import build_eval_env
    from unittest.mock import patch

    model_path = tmp_path / "model.flax"
    patch_target = "brittle_star_project.evaluation.eval_env_builder.PolicyAgent.from_checkpoint"
    with patch(patch_target) as mock_agent:
        mock_agent.return_value = "mock_policy"
        bundle = build_eval_env(
            model_path=model_path,
            training=mock_training_config,
            metadata=mock_metadata,
            morphology_override_path=None,
        )
        assert bundle.segments_per_arm == [1, 1, 1, 1, 1]
        assert bundle.num_active_arms == 5
        assert bundle.architecture == "CENTRALIZED"
        assert bundle.policy == "mock_policy"


def test_build_eval_env_override_morphology(tmp_path, mock_training_config, mock_metadata):
    from brittle_star_project.evaluation.eval_env_builder import build_eval_env
    from unittest.mock import patch

    model_path = tmp_path / "model.flax"
    override_path = tmp_path / "override.yaml"
    override_path.write_text(yaml.dump({"segments_per_arm": [1, 0, 1, 0, 1]}))

    with patch("brittle_star_project.evaluation.eval_env_builder.PolicyAgent.from_checkpoint"):
        bundle = build_eval_env(
            model_path=model_path,
            training=mock_training_config,
            metadata=mock_metadata,
            morphology_override_path=override_path,
        )
        assert bundle.segments_per_arm == [1, 0, 1, 0, 1]
        assert bundle.num_active_arms == 3
        # Should be smaller than 5*N
        assert sum(bundle.action_mask) < len(bundle.action_mask)


def test_build_eval_env_action_mask_shape(tmp_path, mock_training_config, mock_metadata):
    from brittle_star_project.evaluation.eval_env_builder import build_eval_env
    from unittest.mock import patch

    model_path = tmp_path / "model.flax"
    override_path = tmp_path / "override.yaml"
    override_path.write_text(yaml.dump({"segments_per_arm": [1, 0, 1, 0, 0]}))

    with patch("brittle_star_project.evaluation.eval_env_builder.PolicyAgent.from_checkpoint"):
        bundle = build_eval_env(
            model_path=model_path,
            training=mock_training_config,
            metadata=mock_metadata,
            morphology_override_path=override_path,
        )
        # For each segment with P-control, there's 2 actions (pitch and yaw).
        # Total segments = 5 -> 10 actions for training.
        assert len(bundle.action_mask) == 10
        # Active segments = 2 -> 4 actions active.
        assert sum(bundle.action_mask) == 4


def test_build_eval_env_morph_mode_inherited(tmp_path, mock_training_config, mock_metadata):
    from brittle_star_project.evaluation.eval_env_builder import build_eval_env
    from brittle_star_project.environment.env_config import MorphMode
    from unittest.mock import patch

    model_path = tmp_path / "model.flax"
    override_path = tmp_path / "override.yaml"
    # No morph_mode in the override YAML
    override_path.write_text(yaml.dump({"segments_per_arm": [1, 0, 1, 0, 1]}))

    # Change training config to be RING
    mock_training_config.morphology.morph_mode = MorphMode.RING

    with patch("brittle_star_project.evaluation.eval_env_builder.PolicyAgent.from_checkpoint"):
        bundle = build_eval_env(
            model_path=model_path,
            training=mock_training_config,
            metadata=mock_metadata,
            morphology_override_path=override_path,
        )
        assert bundle.architecture == "RING"
