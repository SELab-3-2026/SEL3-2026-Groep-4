import jax
import jax.numpy as jnp
from typing import Dict, Tuple, Optional

from brittle_star_project.environment.env_config import MorphMode

from experiment_logger import get_logger

logger = get_logger()

_JOINT_SCALED_KEYS = frozenset(
    {
        "joint_position",
        "joint_velocity",
        "joint_actuator_force",
        "actuator_force",
    }
)

_SEGMENT_SCALED_KEYS = frozenset(
    {
        "segment_contact",
    }
)


def _build_joint_indices(segments_per_arm, indices_mlp):
    indices = []
    start = 0
    for i, segs in enumerate(segments_per_arm):
        # 2 joints per segment
        if i in indices_mlp:
            count = segs * 2
            idx = jnp.arange(start, start + count)
            indices.append(idx)
            start += count
    return indices


def _build_segment_indices(segments_per_arm, indices_mlp):
    indices = []
    start = 0
    for i, segs in enumerate(segments_per_arm):
        if i in indices_mlp:
            idx = jnp.arange(start, start + segs)
            indices.append(idx)
            start += segs
    return indices


def create_obs_processor(
    bounds_dict: Dict[str, Tuple[float, float]],
    num_arms: int,
    needed_copies: int,
    padding_masks: Optional[Dict] = None,
    morph_mode: MorphMode = MorphMode.CENTRALIZED,
    segments_per_arm=[4, 4, 4, 4, 4],
    agent_indices=[0, 1, 2, 3, 4],
):
    # made a set to allow O(1) search
    ordered_keys = frozenset(
        [
            "disk_z_tilt",
            "joint_actuator_force",
            "joint_position",
            "joint_velocity",
            "robot_direction_to_target",
            "segment_contact",
        ]
    )
    segment_indices = _build_segment_indices(segments_per_arm, agent_indices)
    joint_indices = _build_joint_indices(segments_per_arm, agent_indices)

    def _add_derived_features(obs: dict) -> dict:
        new_obs = dict(obs)
        if "disk_rotation" in new_obs:
            rot = new_obs["disk_rotation"]
            new_obs["disk_z_tilt"] = jnp.sqrt(jnp.pow(rot[0], 2) + jnp.pow(rot[1], 2))

            if "unit_xy_direction_to_target" in new_obs:
                yaw = rot[2]
                unit_x, unit_y = new_obs["unit_xy_direction_to_target"]
                cos_yaw, sin_yaw = jnp.cos(yaw), jnp.sin(yaw)
                new_x = unit_x * cos_yaw + unit_y * sin_yaw
                new_y = -unit_x * sin_yaw + unit_y * cos_yaw
                new_obs["robot_direction_to_target"] = jnp.stack([new_x, new_y])

        return new_obs

    def _normalize_features(obs: dict) -> dict:
        normalized = {}
        for key, arr in obs.items():
            if key in bounds_dict:
                low, high = bounds_dict[key]
                if low == -1.0 and high == 1.0:
                    normalized[key] = jnp.clip(arr, -1.0, 1.0)
                else:
                    arr_clipped = jnp.clip(arr, low, high)
                    normalized[key] = 2.0 * (arr_clipped - low) / (high - low) - 1.0
            else:
                normalized[key] = arr
        return normalized

    def _split_to_agents(obs: dict, morph_mode) -> dict:
        output = {}
        num_agents = needed_copies  # IMPORTANT: number of MLPs

        segs_per_arm = 4
        joints_per_segment = 2
        joints_per_arm = segs_per_arm * joints_per_segment
        for key, arr in obs.items():
            arr = jnp.asarray(arr)
            if arr.size == 0:
                continue

            if arr.ndim == 0:
                arr = arr.reshape(1)

            if key in _SEGMENT_SCALED_KEYS:
                per_agent = []
                for i, _ in enumerate(agent_indices):
                    idx = segment_indices[i]
                    taken = jnp.take(arr, idx, axis=0)
                    pad_len = segs_per_arm - taken.shape[0]
                    padded = jnp.pad(taken, [(0, pad_len)] + [(0, 0)] * (taken.ndim - 1))

                    per_agent.append(padded.reshape(-1))
                arr = jnp.stack(per_agent)
            elif key in _JOINT_SCALED_KEYS:
                per_agent = []
                for i, _ in enumerate(agent_indices):
                    idx = joint_indices[i]
                    taken = jnp.take(arr, idx, axis=0)
                    pad_len = joints_per_arm - taken.shape[0]
                    padded = jnp.pad(taken, [(0, pad_len)] + [(0, 0)] * (taken.ndim - 1))

                    per_agent.append(padded.reshape(-1))
                arr = jnp.stack(per_agent)
            else:
                arr = jnp.repeat(arr[None, :], num_agents, axis=0)

            if morph_mode == MorphMode.CENTRALIZED:
                output[key] = arr.reshape(1, -1)
            elif key in _JOINT_SCALED_KEYS:
                output[key] = arr.reshape(num_agents, -1)
            elif key in _SEGMENT_SCALED_KEYS:
                output[key] = arr[:, None]
            else:
                output[key] = arr

        return output

    def _flatten_features(obs: dict) -> jnp.ndarray:
        """
        Input:
            key -> (num_arms, feat_per_key)

        Output:
            (num_arms, total_features)
        """
        values = []

        for key in sorted(ordered_keys):
            if key not in obs:
                continue

            arr = jnp.asarray(obs[key])  # (num_arms, feat)

            if arr.size == 0:
                continue

            if arr.ndim == 1:
                arr = arr[:, None]

            arr = arr.reshape(arr.shape[0], -1)

            values.append(arr)

        return jnp.concatenate(values, axis=-1)  # (num_arms, total_feat)

    def _process_single(obs_dict: dict) -> jnp.ndarray:
        processed = _add_derived_features(obs_dict)
        processed = _normalize_features(processed)
        processed = _split_to_agents(processed, morph_mode)
        flat = _flatten_features(processed)  # (num_arms, total_feat)

        logger.debug(f"[FLATTENED FINAL] shape: {flat.shape}")
        logger.debug(f"[PER AGENT] example row 0 shape: {flat[0].shape}")

        return flat  # (agents, feat)

    return jax.jit(jax.vmap(_process_single))
