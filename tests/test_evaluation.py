import numpy as np

from brittle_star_project.evaluation.checkpoint import metadata_to_configs, TrainingConfig
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
