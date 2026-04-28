import jax
import jax.numpy as jnp
from typing import Dict, Tuple, Optional

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
    bounds_dict: Dict[str, Tuple[float, float]], padding_masks: Optional[Dict] = None
):
    def _add_derived_features(obs: dict) -> dict:
        new_obs = dict(obs)
        if "disk_rotation" in new_obs:
            rot = new_obs["disk_rotation"]
            new_obs["disk_z_tilt"] = jnp.sqrt(jnp.pow(rot[0], 2) + jnp.pow(rot[1], 2))
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
            "segment_contact",
            "unit_xy_direction_to_target",
        ]
        values = []
        for key in ordered_keys:
            if key in obs:
                arr = jnp.asarray(obs[key]).flatten()
                if arr.size > 0:
                    values.append(arr)
        return jnp.concatenate(values)

    def _process_single(obs_dict: dict) -> jnp.ndarray:
        processed = _add_derived_features(obs_dict)
        processed = _normalize_features(processed)
        if padding_masks is not None:
            processed = _pad_features(processed)
        return _flatten_features(processed)

    return jax.jit(jax.vmap(_process_single))
