import jax.numpy as jnp

from brittle_star_project.configs.main_config import BrittleStarConfig
from brittle_star_project.environment.env_config import MorphMode
from brittle_star_project.environment.env_types import Backend
from brittle_star_project.environment.BrittleStarJaxEnvWrapper import BrittleStarJaxEnvWrapper
from brittle_star_project.environment.obs_processing import create_obs_processor


def test_raw_environment_returns_allocentric_direction():
    """
    Verifies that the raw environment returns a GLOBAL (allocentric)
    direction to the target. If the robot rotates in place,
    the global vector to the target should remain identical.
    """
    env = BrittleStarJaxEnvWrapper.default(num_envs=1, backend=Backend.MJX)
    env_state = env.reset(seed=42)
    raw_obs_1 = env_state.observations["unit_xy_direction_to_target"]

    ninety_deg_z_quat = jnp.array([0.7071068, 0.0, 0.0, 0.7071068])
    new_qpos = env_state.mjx_data.qpos.at[..., 3:7].set(ninety_deg_z_quat)
    new_data = env_state.mjx_data.replace(qpos=new_qpos)
    rotated_env_state = env_state.replace(mjx_data=new_data)

    zero_action = jnp.zeros(env.single_action_space.shape)
    if len(raw_obs_1.shape) > 1:
        zero_action = jnp.expand_dims(zero_action, 0)
    final_env_state = env.step(rotated_env_state, zero_action)
    raw_obs_2 = final_env_state.observations["unit_xy_direction_to_target"]

    # If the vector is allocentric, it should not change when the robot spins.
    assert jnp.sum(jnp.abs(raw_obs_1 - raw_obs_2)) < 1e-4, (
        f"The raw environment observation changed when the robot rotated! "
        f"This means it is already egocentric. "
        f"Obs 1: {raw_obs_1}, Obs 2: {raw_obs_2}"
    )


def test_processor_converts_to_egocentric_direction():
    """
    Verifies that the obs_processor correctly applies a 2D inverse rotation
    matrix to convert the global target vector into a local (egocentric) vector.
    """
    cfg = BrittleStarConfig()
    env = BrittleStarJaxEnvWrapper.default(num_envs=1, backend=Backend.MJX)

    segments_per_arm = jnp.array((4, 4, 4, 4, 4))
    num_arms = jnp.where(segments_per_arm > 0, 1, 0).sum().item()

    obs_processor = create_obs_processor(
        bounds_dict=cfg.obs_bounds.to_bounds_dict(),
        needed_copies=1,
        num_arms=num_arms,
        padding_masks=env.padding_masks,
        morph_mode=MorphMode.CENTRALIZED,
    )

    env_state = env.reset(seed=42)

    # --- Scenario 1 ---
    # Robot is rotated 90 degrees Left (facing global Y)
    # Target is straight ahead on the global X axis [1.0, 0.0]
    # Because the robot is facing Y, the target on X is to its RIGHT [0.0, -1.0] locally.
    dummy_obs_1 = dict(env_state.observations)
    dummy_obs_1["disk_rotation"] = jnp.array([[0.0, 0.0, jnp.pi / 2.0]])
    dummy_obs_1["unit_xy_direction_to_target"] = jnp.array([[1.0, 0.0]])
    processed_1 = obs_processor(dummy_obs_1)

    # --- Scenario 2 (used to find the array indices) ---
    # We change ONLY the target vector so we can isolate it in the final array
    dummy_obs_2 = dict(env_state.observations)
    dummy_obs_2["disk_rotation"] = jnp.array([[0.0, 0.0, jnp.pi / 2.0]])
    dummy_obs_2["unit_xy_direction_to_target"] = jnp.array([[0.0, 1.0]])
    processed_2 = obs_processor(dummy_obs_2)

    # Find the indices of the elements that changed
    diff_array = jnp.abs(processed_1[0, 0] - processed_2[0, 0])
    changed_indices = jnp.where(diff_array > 1e-4)[0]

    # (143,)
    local_target = processed_1[0, 0, changed_indices]

    # (2,)
    expected_local_target = jnp.array([0.0, -1.0])

    assert jnp.sum(jnp.abs(local_target - expected_local_target)) < 1e-4, (
        f"The obs_processor did not correctly rotate the vector to egocentric. "
        f"Expected {expected_local_target}, but got {local_target}."
    )
