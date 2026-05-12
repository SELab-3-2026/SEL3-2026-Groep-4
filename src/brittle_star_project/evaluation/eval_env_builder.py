from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import jax.numpy as jnp
import numpy as np
import yaml
from omegaconf import OmegaConf

from brittle_star_project import Backend, BrittleStarEnv, BrittleStarEnvFactory
from brittle_star_project.environment.env_config import MorphMode, MorphologyConfig
from brittle_star_project.environment.obs_processing import create_obs_processor
from brittle_star_project.environment.padded_obs_wrapper import compute_padding_masks
from brittle_star_project.evaluation.checkpoint import TrainingConfig
from brittle_star_project.evaluation.policy import PolicyAgent
from brittle_star_project.MLPs.adjancency_builder import build_adjacency


@dataclass
class EvalEnvBundle:
    """Everything needed to run a headless evaluation episode."""

    env: BrittleStarEnv
    policy: PolicyAgent
    action_low: np.ndarray | None
    action_high: np.ndarray | None
    action_mask: np.ndarray | None
    segments_per_arm: list[int]
    num_active_arms: int
    architecture: str


def build_eval_env(
    *,
    model_path: Path,
    training: TrainingConfig,
    metadata: dict,
    morphology_override_path: Path | str | None = None,
) -> EvalEnvBundle:
    """Build environment + policy for evaluation, optionally with a morphology override."""

    # 1. Determine environment morphology
    if morphology_override_path is not None:
        override_path = Path(morphology_override_path)
        if not override_path.exists():
            raise FileNotFoundError(f"Could not find morphology override YAML at {override_path}")
        with open(override_path, "r") as f:
            override_dict = yaml.safe_load(f)
        env_morphology = OmegaConf.to_object(
            OmegaConf.merge(OmegaConf.structured(MorphologyConfig), override_dict)
        )
        # Force morph_mode to be inherited from training since it's baked into weights
        env_morphology.morph_mode = training.morphology.morph_mode
    else:
        env_morphology = training.morphology

    # 2. Build obs_processor with TRAINING morphology padding masks always
    padding_masks = compute_padding_masks(
        segments_per_arm=env_morphology.segments_per_arm,
        reference_segments_per_arm=training.morphology.segments_per_arm,
    )

    training_segs_per_arm = jnp.array(training.morphology.segments_per_arm)

    needed_copies = 0
    agent_indices = [0, 1, 2, 3, 4]
    match training.morphology.morph_mode:
        case MorphMode.CENTRALIZED:
            needed_copies = 1
        case MorphMode.FULLY_CONNECTED | MorphMode.RING:
            agent_mask = training_segs_per_arm > 0
            agent_indices = jnp.where(agent_mask)[0].tolist()
            needed_copies = jnp.where(training_segs_per_arm > 0, 1, 0).sum().item()
        case MorphMode.SEGMENT:
            agent_mask = training_segs_per_arm > 0
            agent_indices = jnp.where(agent_mask)[0].tolist()
            needed_copies = (
                training_segs_per_arm.sum() + jnp.where(training_segs_per_arm > 0, 1, 0).sum()
            ).item()

    num_arms_training = jnp.where(training_segs_per_arm > 0, 1, 0).sum().item()

    obs_processor = create_obs_processor(
        bounds_dict=training.obs_bounds.to_bounds_dict(),
        padding_masks=padding_masks,
        needed_copies=needed_copies,
        num_arms=num_arms_training,
        morph_mode=training.morphology.morph_mode,
        segments_per_arm=env_morphology.segments_per_arm,
        agent_indices=agent_indices,
    )

    # 3. Build environment
    backend = Backend.MJC
    factory = BrittleStarEnvFactory()
    raw_env = factory.create_environment(
        backend,
        env_morphology,
        training.arena,
        training.environment,
    )
    env = BrittleStarEnv(
        raw_env,
        backend=backend,
        config=training.environment,
        morphology_config=env_morphology,
    )

    # Calculate the action dimension the model was trained with
    training_total_actions = sum(training.morphology.segments_per_arm) * 2
    trained_action_dim = training_total_actions // needed_copies

    # 4. Load policy
    message_passing_steps = (metadata.get("architecture", {}) or {}).get("message_passing_steps")
    if message_passing_steps is None:
        message_passing_steps = 4
    message_passing_steps = int(message_passing_steps)

    adj_matrix = None
    if training.morphology.morph_mode != MorphMode.CENTRALIZED:
        adj_matrix = build_adjacency(
            training.morphology.segments_per_arm, training.morphology.morph_mode
        )

        override_segs = env_morphology.segments_per_arm
        if training.morphology.morph_mode in (MorphMode.FULLY_CONNECTED, MorphMode.RING):
            for i, segs in enumerate(override_segs):
                if segs == 0 and i < adj_matrix.shape[0]:
                    adj_matrix = adj_matrix.at[i, :].set(0)
                    adj_matrix = adj_matrix.at[:, i].set(0)
        elif training.morphology.morph_mode == MorphMode.SEGMENT:
            for i, segs in enumerate(override_segs):
                if segs == 0 and i < num_arms_training:
                    adj_matrix = adj_matrix.at[i, :].set(0)
                    adj_matrix = adj_matrix.at[:, i].set(0)

            idx = 0
            for arm_idx, seg_count in enumerate(training.morphology.segments_per_arm):
                if override_segs[arm_idx] == 0:
                    for i in range(seg_count):
                        seg_node = num_arms_training + idx + i
                        if seg_node < adj_matrix.shape[0]:
                            adj_matrix = adj_matrix.at[seg_node, :].set(0)
                            adj_matrix = adj_matrix.at[:, seg_node].set(0)
                idx += seg_count

    policy = PolicyAgent.from_checkpoint(
        model_path,
        action_dim=trained_action_dim,
        obs_processor=obs_processor,
        message_passing_steps=message_passing_steps,
        adj_matrix=adj_matrix,
    )

    # 5. Build action clipping and masks
    action_mask = np.asarray(padding_masks["mask_2x"])

    action_space = getattr(raw_env, "action_space", None)
    action_low = (
        None if action_space is None else np.asarray(action_space.low, dtype=np.float32).ravel()
    )
    action_high = (
        None if action_space is None else np.asarray(action_space.high, dtype=np.float32).ravel()
    )

    return EvalEnvBundle(
        env=env,
        policy=policy,
        action_low=action_low,
        action_high=action_high,
        action_mask=action_mask,
        segments_per_arm=env_morphology.segments_per_arm,
        num_active_arms=sum(1 for s in env_morphology.segments_per_arm if s > 0),
        architecture=env_morphology.morph_mode.name,
    )
