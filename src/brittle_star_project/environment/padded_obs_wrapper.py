"""Observation padding wrapper for amputated brittle star morphologies.

When using a centralized controller, the global observation vector must remain
a constant size regardless of how many segments are amputated. This wrapper pads
the observation dictionary values with zeros using spatial insertion so that the
flattened observation maintains the correct physical mapping to the neural network.
"""
from __future__ import annotations

from typing import Any
import jax.numpy as jnp

# Observation keys whose size scales with the number of joints (2 per segment).
_JOINT_SCALED_KEYS = frozenset({
    "joint_position",
    "joint_velocity",
    "joint_actuator_force",
    "actuator_force",
})

# Observation keys whose size scales with the number of segments (1 per segment).
_SEGMENT_SCALED_KEYS = frozenset({
    "segment_contact",
})


def compute_padding_masks(
    segments_per_arm: tuple[int, ...],
    reference_segments_per_arm: tuple[int, ...] = (4, 4, 4, 4, 4),
) -> dict[str, Any]:
    """Pre-compute boolean masks for spatial insertion of observations.

    Args:
        segments_per_arm: The current (possibly amputated) morphology.
        reference_segments_per_arm: The full morphology that defines the expected size.

    Returns:
        A dict containing 1D boolean masks and target sizes.
    """
    mask_1x = []
    mask_2x = []

    for actual, ref in zip(segments_per_arm, reference_segments_per_arm):
        # 1x scaling (e.g., contacts: 1 value per segment)
        mask_1x.extend([True] * actual + [False] * (ref - actual))
        # 2x scaling (e.g., joints: 2 values per segment)
        mask_2x.extend([True] * (actual * 2) + [False] * ((ref - actual) * 2))

    return {
        "mask_1x": jnp.array(mask_1x, dtype=bool),
        "mask_2x": jnp.array(mask_2x, dtype=bool),
        "target_size_1x": sum(reference_segments_per_arm),
        "target_size_2x": sum(reference_segments_per_arm) * 2,
    }


def pad_observation(
    obs: dict[str, Any],
    masks: dict[str, Any],
) -> dict[str, Any]:
    """Pad an observation dict using spatial insertion."""
    padded = {}
    for key, value in obs.items():
        if key in _JOINT_SCALED_KEYS:
            out = jnp.zeros(masks["target_size_2x"], dtype=value.dtype)
            padded[key] = out.at[masks["mask_2x"]].set(value)
        elif key in _SEGMENT_SCALED_KEYS:
            out = jnp.zeros(masks["target_size_1x"], dtype=value.dtype)
            padded[key] = out.at[masks["mask_1x"]].set(value)
        else:
            padded[key] = value
    return padded


def pad_observations_batched(
    obs: dict[str, Any],
    masks: dict[str, Any],
) -> dict[str, Any]:
    """Pad a batched observation dict (leading batch dimension) using spatial insertion."""
    padded = {}
    for key, value in obs.items():
        batch_size = value.shape[0]
        if key in _JOINT_SCALED_KEYS:
            out = jnp.zeros((batch_size, masks["target_size_2x"]), dtype=value.dtype)
            padded[key] = out.at[:, masks["mask_2x"]].set(value)
        elif key in _SEGMENT_SCALED_KEYS:
            out = jnp.zeros((batch_size, masks["target_size_1x"]), dtype=value.dtype)
            padded[key] = out.at[:, masks["mask_1x"]].set(value)
        else:
            padded[key] = value
    return padded
