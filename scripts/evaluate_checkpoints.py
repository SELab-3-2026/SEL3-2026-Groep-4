"""Re-evaluate saved checkpoints from a completed training run using MJX.

This script scans the checkpoint directory of a training run (the `checkpoints/`
folder inside a Hydra output directory), loads each `.flax` checkpoint, runs
one deterministic evaluation episode with `build_eval_rollout_fn`, and appends
the result to the run's `metrics/checkpoint_evaluation.csv`.

It is intended for post-training analysis when per-checkpoint evaluation was not
enabled during training (`evaluate_checkpoints: false`).

Usage:
    python scripts/evaluate_checkpoints.py \
        simulation.model_path=runs/2024-01-01/12-00-00/final_model.flax \
        evaluation.eval_max_steps=5000 \
        evaluation.eval_seed=0

The script resolves the run directory from `simulation.model_path`, discovers
all `*.flax` checkpoints under `checkpoints/`, and evaluates them in order.
"""

from __future__ import annotations
from brittle_star_project.MLPs.mlps import (
    Actor,
    GenericDenseLayersWithActivation,
    MessagePasser,
)
from brittle_star_project.MLPs.adjancency_builder import build_adjacency
from brittle_star_project.environment import MorphMode
from brittle_star_project.trainers.PPOTrainer import apply_per_node
import logging
import re
from pathlib import Path

import hydra
import jax
import numpy as np
import jax.numpy as jnp

from omegaconf import DictConfig, OmegaConf

from brittle_star_project.configs.main_config import BrittleStarConfig
from brittle_star_project.configs.register_configs import register_configs
from brittle_star_project.environment.BrittleStarJaxEnvWrapper import BrittleStarJaxEnvWrapper
from brittle_star_project.environment.obs_processing import create_obs_processor
from brittle_star_project.environment.padded_obs_wrapper import compute_padding_masks
from brittle_star_project.evaluation.checkpoint import (
    load_metadata,
    load_params,
    metadata_to_configs,
)
from brittle_star_project.evaluation.evaluate_mjx import (
    append_checkpoint_eval_row,
    build_eval_rollout_fn,
    evaluate_checkpoint_mjx,
)
from brittle_star_project.trainers.PPOTrainer import reward_fn


def _parse_iteration(checkpoint_path: Path) -> int:
    """Parse the iteration number from a checkpoint filename like `checkpoint_0042.flax`."""
    match = re.search(r"(\d+)", checkpoint_path.stem)
    return int(match.group(1)) if match else -1


@hydra.main(config_path="../configs", config_name="main_config", version_base="1.3")
def main(dict_cfg: DictConfig) -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    logger = logging.getLogger(__name__)

    cfg: BrittleStarConfig = OmegaConf.to_object(
        OmegaConf.merge(OmegaConf.structured(BrittleStarConfig), dict_cfg)
    )
    sim_cfg = cfg.simulation
    eval_cfg = cfg.evaluation

    # --- Resolve the model path to find the run directory ---
    model_path_str = sim_cfg.model_path
    if model_path_str is None:
        raise ValueError(
            "simulation.model_path must point to the final_model.flax of a training run."
        )

    model_path = Path(hydra.utils.to_absolute_path(model_path_str))
    run_dir = model_path.parent

    checkpoints_dir = run_dir / "checkpoints"
    if not checkpoints_dir.exists():
        raise FileNotFoundError(
            f"No checkpoints/ directory found in run directory: {run_dir}\n"
            "Make sure simulation.model_path points to a completed training run."
        )

    checkpoints = sorted(checkpoints_dir.glob("*.flax"), key=_parse_iteration)
    if not checkpoints:
        raise FileNotFoundError(f"No .flax checkpoints found in {checkpoints_dir}")

    logger.info(f"Found {len(checkpoints)} checkpoint(s) in {checkpoints_dir}")

    # --- Load sidecar metadata + reconstruct training config ---
    metadata_override = (
        Path(hydra.utils.to_absolute_path(sim_cfg.metadata_path))
        if sim_cfg.metadata_path is not None
        else None
    )
    metadata = load_metadata(model_path, metadata_override)
    training = metadata_to_configs(metadata)

    padding_masks = compute_padding_masks(
        segments_per_arm=training.morphology.segments_per_arm,
        reference_segments_per_arm=training.morphology.segments_per_arm,
    )

    morph_mode = training.morphology.morph_mode

    segments_per_arm = jnp.asarray(
        training.morphology.segments_per_arm,
        dtype=jnp.int32,
    )

    num_arms = (
        jnp.where(
            segments_per_arm > 0,
            1,
            0,
        )
        .sum()
        .item()
    )

    match morph_mode:
        case MorphMode.CENTRALIZED:
            needed_copies = 1
            agent_indices = [0, 1, 2, 3, 4]

        case MorphMode.FULLY_CONNECTED | MorphMode.RING:
            agent_mask = segments_per_arm > 0
            agent_indices = jnp.where(agent_mask)[0]
            needed_copies = num_arms

        case MorphMode.SEGMENT:
            agent_mask = segments_per_arm > 0
            agent_indices = jnp.where(agent_mask)[0]

            needed_copies = (segments_per_arm.sum() + num_arms).item()

    obs_processor = create_obs_processor(
        bounds_dict=training.obs_bounds.to_bounds_dict(),
        padding_masks=padding_masks,
        num_arms=num_arms,
        needed_copies=needed_copies,
        morph_mode=morph_mode,
        segments_per_arm=segments_per_arm,
        agent_indices=agent_indices,
    )

    env = BrittleStarJaxEnvWrapper(
        morphology=training.morphology,
        arena=training.arena,
        env_config=training.environment,
        num_envs=1,
    )

    action_low = np.asarray(env.single_action_space.low, dtype=np.float32)
    action_high = np.asarray(env.single_action_space.high, dtype=np.float32)

    sensor = GenericDenseLayersWithActivation(layer_sizes=[300, 300, 300])
    actor = Actor(action_dim=env.single_action_space.shape[0])
    sensor.apply = jax.jit(sensor.apply)
    actor.apply = jax.jit(actor.apply)

    eval_fn = build_eval_rollout_fn(
        env=env,
        obs_processor=obs_processor,
        sensor_apply=sensor.apply,
        actor_apply=actor.apply,
        action_low=action_low,
        action_high=action_high,
        reward_fn=reward_fn,
    )

    morph_mode = training.morphology.morph_mode

    segments_per_arm = jnp.asarray(
        training.morphology.segments_per_arm,
        dtype=jnp.int32,
    )

    match morph_mode:
        case MorphMode.CENTRALIZED:
            needed_copies = 1

        case MorphMode.FULLY_CONNECTED | MorphMode.RING:
            needed_copies = jnp.where(segments_per_arm > 0, 1, 0).sum().item()

        case MorphMode.SEGMENT:
            needed_copies = (
                segments_per_arm.sum() + jnp.where(segments_per_arm > 0, 1, 0).sum()
            ).item()

    adj = build_adjacency(
        training.morphology.segments_per_arm,
        morph_mode,
    )

    sensor = GenericDenseLayersWithActivation(layer_sizes=[300, 300, 300])

    actor = Actor(action_dim=env.single_action_space.shape[0] // needed_copies)

    message_passer = (
        MessagePasser(
            hidden_dim=300,
            num_propagation_steps=4,
            adj_matrix=adj,
        )
        if morph_mode != MorphMode.CENTRALIZED
        else None
    )

    eval_fn = build_eval_rollout_fn(
        env=env,
        obs_processor=obs_processor,
        sensor_apply=lambda p, x: apply_per_node(sensor, p, x),
        actor_apply=lambda p, x: apply_per_node(actor, p, x),
        message_passer_apply=(None if message_passer is None else message_passer.apply),
        action_low=action_low,
        action_high=action_high,
        reward_fn=reward_fn,
    )
    seed = int(eval_cfg.eval_seed)
    max_steps = int(eval_cfg.eval_max_steps)

    logger.info(f"Evaluating each checkpoint (seed={seed}, max_steps={max_steps}).")

    for checkpoint_path in checkpoints:
        iteration = _parse_iteration(checkpoint_path)
        try:
            params = load_params(checkpoint_path)
        except Exception as e:
            logger.warning(f"Could not load {checkpoint_path.name}: {e}")
            continue

        result = evaluate_checkpoint_mjx(eval_fn, params, seed=seed, max_steps=max_steps)
        csv_path = append_checkpoint_eval_row(
            run_dir,
            iteration=iteration,
            trained_timesteps=0,  # unknown without training logs
            result=result,
        )

        logger.debug(
            f"checkpoint={iteration:5d} | "
            f"reached={str(result.reached_target):<5} | "
            f"return={result.eval_return:+8.3f} | "
            f"steps={result.steps:4d} | "
            f"final_dist={result.final_xy_dist:.3f}"
        )

    logger.info(f"Done. CSV at: {csv_path}")
    env.close()


if __name__ == "__main__":
    register_configs()
    main()
