import jax
import jax.numpy as jnp
from typing import Dict, Tuple, Optional

from brittle_star_project.environment.env_config import MorphMode

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


def create_obs_processor(
    bounds_dict: Dict[str, Tuple[float, float]],
    num_segments: int,
    num_arms: int,
    padding_masks: Optional[Dict] = None,
    morph_mode: MorphMode = MorphMode.CENTRALIZED,
):
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

    def _pad_features(obs: dict) -> dict:
        assert padding_masks is not None

        padded = {}
        for key, arr in obs.items():
            if key in _JOINT_SCALED_KEYS:
                padded_arr = jnp.zeros(padding_masks["target_size_2x"], dtype=arr.dtype)
                padded[key] = padded_arr.at[padding_masks["mask_2x"]].set(arr)
            elif key in _SEGMENT_SCALED_KEYS:
                padded_arr = jnp.zeros(padding_masks["target_size_1x"], dtype=arr.dtype)
                padded[key] = padded_arr.at[padding_masks["mask_1x"]].set(arr)
            else:
                padded[key] = arr
        return padded

    def _flatten_features(obs: dict) -> jnp.ndarray:
        ordered_keys = [
            "disk_z_tilt",
            "joint_actuator_force",
            "joint_position",
            "joint_velocity",
            "robot_direction_to_target",
            "segment_contact",
        ]

        values = []

        for key in sorted(obs.keys()):
            if key not in ordered_keys:
                continue
            v = obs[key]

            # skip empty arrays and scalars
            if v.size == 0:
                continue

            # reshape scalars
            if v.ndim == 0:
                v = v.reshape(1)

            # -------- CENTRALIZED --------
            if morph_mode == MorphMode.CENTRALIZED:
                values.append(v.reshape(1, -1))  # (1, feat)
                continue

            # -------- SPLIT TO SEGMENTS --------
            if key in _JOINT_SCALED_KEYS:
                if morph_mode == MorphMode.SEGMENT:
                    center_size = num_arms * 3 * 2
                    v_center = v[:center_size].reshape(num_arms, 3 * 2)  # (arms, 6)
                    v_segs = v[center_size:].reshape(-1, 2)  # (segs, 2)
                    values.append(jnp.concatenate([v_center, v_segs], axis=0))  # (arms+segs, ?)
                    continue
                v = v.reshape(num_arms, -1)  # (n_arms, 2)

            elif key in _SEGMENT_SCALED_KEYS:
                v = v[:, None]  # (segments, 1)

            else:
                # global key, broadcast to all nodes
                n_nodes = (num_segments + num_arms) if morph_mode == MorphMode.SEGMENT else num_arms
                v = jnp.repeat(v[None, :], n_nodes, axis=0)  # (n_nodes, feat)

            # -------- SEGMENT MODE --------
            if morph_mode == MorphMode.SEGMENT:
                values.append(v)  # (n_nodes, feat)
                continue

            # -------- ARM MODE --------
            v = v.reshape(num_arms, -1)
            values.append(v)  # (n_arms, feat)

        return jnp.concatenate(values, axis=-1)

    def _process_single(obs_dict: dict) -> jnp.ndarray:
        processed = _add_derived_features(obs_dict)
        processed = _normalize_features(processed)
        if padding_masks is not None:
            processed = _pad_features(processed)
        return _flatten_features(processed)

    return jax.jit(jax.vmap(_process_single))
