import jax
import jax.numpy as jnp

from brittle_star_project.environment.obs_processing import create_obs_processor
from brittle_star_project.environment.env_config import MorphMode, ObservationBoundsConfig


obs_bounds = ObservationBoundsConfig().to_bounds_dict()

"""
Test for obs_processor.

Centralized: 40 features per agent:
    disk_z_tilt              → scalar → reshaped to (1,)  → 1 feat
    joint_actuator_force     → 8 joints padded to 8       → 8 feat
    joint_position           → 8 joints padded to 8       → 8 feat
    joint_velocity           → 8 joints padded to 8       → 8 feat
    robot_direction_to_target→ (x, y)                     → 2 feat
    segment_contact          → 4 segs, pre-padded by 9    → 13 feat (9 leading + 4)
"""

NUM_ARMS = 5
SEGS_PER_ARM = 4  # healthy segments per arm
JOINTS_PER_SEG = 2  # from _build_joint_indices: segs * 2

SEGS_HEALTHY = [4, 4, 4, 4, 4]
SEGS_DAMAGED = [4, 4, 4, 4, 0]  # arm 4 fully disabled
AGENT_INDICES = [0, 1, 2, 3, 4]

FEAT_PER_AGENT = 1 + 8 + 8 + 8 + 2 + 13  # = 40


def make_obs(segs_per_arm: list[int]) -> dict:
    total_segs = sum(segs_per_arm)
    total_joints = JOINTS_PER_SEG * total_segs

    return {
        "actuator_force": jnp.ones(total_joints),
        "disk_angular_velocity": jnp.zeros(3),
        "disk_linear_velocity": jnp.zeros(3),
        "disk_position": jnp.zeros(3),
        "disk_rotation": jnp.array([0.1, 0.1, 0.5]),  # (roll, pitch, yaw)
        "joint_actuator_force": jnp.full(total_joints, 1.0),
        "joint_position": jnp.full(total_joints, 0.5),
        "joint_velocity": jnp.full(total_joints, 2.0),
        "segment_contact": jnp.ones(total_segs),
        "tendon_position": jnp.zeros(0),
        "tendon_velocity": jnp.zeros(0),
        "unit_xy_direction_to_target": jnp.array([1.0, 0.0]),
        "xy_distance_to_target": jnp.array([3.5]),
    }


def batch_obs(obs: dict):
    return jax.tree_util.tree_map(lambda x: x[None, :], obs)


def make_processor(morph_mode: MorphMode, needed_copies: int, segments_per_arm: list[int]):
    return create_obs_processor(
        bounds_dict=obs_bounds,
        num_arms=NUM_ARMS,
        needed_copies=needed_copies,
        morph_mode=morph_mode,
        segments_per_arm=segments_per_arm,
        agent_indices=AGENT_INDICES,
    )


def test_centralized_no_damage():
    proc = make_processor(MorphMode.CENTRALIZED, 1, SEGS_HEALTHY)
    obs = make_obs(SEGS_HEALTHY)
    obs = batch_obs(obs)
    global_state = proc(obs)

    # shape test
    assert global_state.shape == (1, 1, FEAT_PER_AGENT * NUM_ARMS)

    # TODO: more?


def test_centralized_damaged_1_arm():
    proc = make_processor(MorphMode.CENTRALIZED, 1, SEGS_DAMAGED)
    obs = make_obs(SEGS_DAMAGED)
    obs = batch_obs(obs)
    global_state = proc(obs)

    # shape test
    assert global_state.shape == (1, 1, FEAT_PER_AGENT * NUM_ARMS)


def test_decentralized_fully_connected_no_damage():
    proc = make_processor(MorphMode.FULLY_CONNECTED, NUM_ARMS, SEGS_HEALTHY)
    obs = make_obs(SEGS_HEALTHY)
    obs = batch_obs(obs)
    global_state = proc(obs)

    # shape test
    assert global_state.shape == (1, NUM_ARMS, FEAT_PER_AGENT)


def test_decentralized_fully_connected_damaged_1_arm():
    proc = make_processor(MorphMode.FULLY_CONNECTED, NUM_ARMS, SEGS_DAMAGED)
    obs = make_obs(SEGS_DAMAGED)
    obs = batch_obs(obs)
    global_state = proc(obs)

    # shape test
    assert global_state.shape == (1, NUM_ARMS, FEAT_PER_AGENT)
