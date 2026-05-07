"""MJC-based (CPU) checkpoint evaluation.

This module provides the CPU-bound evaluation path using the standard MJC backend.
It is primarily used by the `evaluate_checkpoints` CLI to compute metrics and
render videos.
"""

from pathlib import Path

import numpy as np

from brittle_star_project.environment.BrittleStarJaxEnvWrapper import BrittleStarJaxEnvWrapper
from brittle_star_project.environment.obs_processing import create_obs_processor
from brittle_star_project.evaluation.policy import PolicyAgent
from brittle_star_project.evaluation.rollout import EpisodeResult, rollout_headless


def evaluate_policy(
    env: BrittleStarJaxEnvWrapper,
    policy_path: str | Path,
    seed: int,
    max_steps: int,
) -> EpisodeResult:
    """Evaluate a trained policy in a CPU-bound environment.

    Args:
        env: Initialised CPU environment (MJC backend).
        policy_path: Path to the ``.cleanrl_model`` weights file.
        seed: Random seed for environment reset.
        max_steps: Maximum number of control steps.

    Returns:
        Structured result containing return, length, and distance metrics.
    """
    obs_processor = create_obs_processor(
        bounds_dict=env.cfg.obs_bounds.to_bounds_dict(),
        padding_masks=env.padding_masks,
    )

    action_dim = env.single_action_space.shape[0]

    policy = PolicyAgent.from_checkpoint(
        model_path=Path(policy_path),
        action_dim=action_dim,
        obs_processor=obs_processor,
    )

    action_low = np.asarray(env.single_action_space.low, dtype=np.float32)
    action_high = np.asarray(env.single_action_space.high, dtype=np.float32)

    return rollout_headless(
        env=env,
        policy=policy,
        seed=seed,
        max_steps=max_steps,
        action_low=action_low,
        action_high=action_high,
    )
