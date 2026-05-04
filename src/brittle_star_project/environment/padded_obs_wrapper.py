"""Observation padding masks for amputated brittle star morphologies."""

from __future__ import annotations

from typing import Any, Sequence

import jax.numpy as jnp


def compute_padding_masks(
    segments_per_arm: Sequence[int],
    reference_segments_per_arm: Sequence[int] = (4, 4, 4, 4, 4),
) -> dict[str, Any]:
    """Pre-compute boolean masks for spatial insertion of observations.

    Args:
        segments_per_arm: The current (possibly amputated) morphology.
        reference_segments_per_arm: The full morphology that defines the expected size.

    Returns:
        A dict containing 1D boolean masks and target sizes.
    """
    if len(segments_per_arm) != len(reference_segments_per_arm):
        raise ValueError(
            f"Morphology mismatch: current has {len(segments_per_arm)} arms, "
            f"but reference requires {len(reference_segments_per_arm)} arms."
        )

    mask_1x = []
    mask_2x = []

    for arm_idx, (actual, ref) in enumerate(zip(segments_per_arm, reference_segments_per_arm)):
        if not (0 <= actual <= ref):
            raise ValueError(
                f"Invalid amputation at arm {arm_idx}: "
                f"actual segments ({actual}) must be between 0 and reference ({ref})."
            )
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
