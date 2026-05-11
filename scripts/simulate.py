"""Simulate a trained policy in the MuJoCo viewer.

Automatically extracts the training configuration (morphology, environment, etc.)
from the sidecar metadata YAML file to ensure simulation perfectly matches training.
Override simulation settings via CLI, e.g.:
    uv run scripts/simulate.py \
        simulation.morphology_override=configs/morphology/3_arms.yaml \
        simulation.model_path=runs/.../final_model.flax
"""

from __future__ import annotations

from pathlib import Path

import hydra
import numpy as np
from omegaconf import DictConfig, OmegaConf
import yaml

import jax.numpy as jnp

from brittle_star_project import Backend, BrittleStarEnv, BrittleStarEnvFactory
from brittle_star_project.configs.main_config import BrittleStarConfig
from brittle_star_project.configs.register_configs import register_configs
from brittle_star_project.environment.padded_obs_wrapper import compute_padding_masks
from brittle_star_project.environment.obs_processing import create_obs_processor
from brittle_star_project.environment.env_config import MorphMode, MorphologyConfig

from brittle_star_project.evaluation.checkpoint import load_metadata, metadata_to_configs
from brittle_star_project.evaluation.policy import PolicyAgent
from brittle_star_project.evaluation.rollout import rollout_headless, rollout_viewer
from brittle_star_project.evaluation.video import (
    record_episode,
    create_evaluation_dir,
    save_evaluation_metadata,
)
from brittle_star_project.MLPs.adjancency_builder import build_adjacency


@hydra.main(config_path="../configs", config_name="main_config", version_base="1.3")
def main(dict_cfg: DictConfig) -> None:
    # 1. Hydra composes ONLY SimulationSettings
    cfg = OmegaConf.to_object(OmegaConf.merge(OmegaConf.structured(BrittleStarConfig), dict_cfg))
    sim_cfg = cfg.simulation

    model_path_str = sim_cfg.model_path
    if model_path_str is None:
        raise ValueError(
            "simulation.model_path must be set to a .flax checkpoint (e.g. final_model.flax)"
        )

    model_path = Path(hydra.utils.to_absolute_path(model_path_str))
    if model_path.suffix != ".flax":
        raise ValueError(f"Expected a '.flax' checkpoint, got '{model_path.name}'.")

    # 2. Discover + load sidecar metadata YAML
    metadata_override = None
    if sim_cfg.metadata_path is not None:
        metadata_override = Path(hydra.utils.to_absolute_path(sim_cfg.metadata_path))

    metadata = load_metadata(model_path, metadata_override)

    # 3. Reconstruct typed configs from metadata
    training = metadata_to_configs(metadata)

    # 4. Determine environment morphology
    if sim_cfg.morphology_override is not None:
        override_path = Path(hydra.utils.to_absolute_path(sim_cfg.morphology_override))
        if not override_path.exists():
            raise FileNotFoundError(f"Could not find morphology override YAML at {override_path}")
        with open(override_path, "r") as f:
            override_dict = yaml.safe_load(f)
        env_morphology = OmegaConf.to_object(
            OmegaConf.merge(OmegaConf.structured(MorphologyConfig), override_dict)
        )
    else:
        env_morphology = training.morphology

    # 5. Build obs_processor with TRAINING morphology padding masks always
    padding_masks = compute_padding_masks(
        segments_per_arm=env_morphology.segments_per_arm,
        reference_segments_per_arm=training.morphology.segments_per_arm,
    )

    segs_per_arm = jnp.array(env_morphology.segments_per_arm)

    needed_copies = 0
    agent_indices = [0, 1, 2, 3, 4]
    match env_morphology.morph_mode:
        case MorphMode.CENTRALIZED:
            needed_copies = 1
        case MorphMode.FULLY_CONNECTED | MorphMode.RING:
            agent_mask = segs_per_arm > 0
            agent_indices = jnp.where(agent_mask)[0]
            needed_copies = jnp.where(segs_per_arm > 0, 1, 0).sum().item()
        case MorphMode.SEGMENT:
            agent_mask = segs_per_arm > 0
            agent_indices = jnp.where(agent_mask)[0]
            needed_copies = jnp.where(segs_per_arm > 0, 1, 0).sum().item()
            needed_copies = (segs_per_arm.sum() + jnp.where(segs_per_arm > 0, 1, 0).sum()).item()

    num_arms = jnp.where(segs_per_arm > 0, 1, 0).sum().item()

    obs_processor = create_obs_processor(
        bounds_dict=training.obs_bounds.to_bounds_dict(),
        padding_masks=padding_masks,
        needed_copies=needed_copies,
        num_arms=num_arms,
        morph_mode=env_morphology.morph_mode,
        segments_per_arm=env_morphology.segments_per_arm,
        agent_indices=agent_indices,
    )

    # 6. Build environment
    backend = Backend.MJC
    seed = int(cfg.experiment.seed)

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

    state0 = env.reset(seed=seed)

    # Calculate the action dimension the model was trained with
    trained_action_dim = raw_env.action_space.shape[0] // needed_copies

    # 7. Load policy
    message_passing_steps = (metadata.get("architecture", {}) or {}).get("message_passing_steps")
    if message_passing_steps is None:
        message_passing_steps = 4
    message_passing_steps = int(message_passing_steps)

    adj_matrix = None
    if env_morphology.morph_mode != MorphMode.CENTRALIZED:
        adj_matrix = build_adjacency(env_morphology.segments_per_arm, env_morphology.morph_mode)

    policy = PolicyAgent.from_checkpoint(
        model_path,
        action_dim=trained_action_dim,
        obs_processor=obs_processor,
        message_passing_steps=message_passing_steps,
        adj_matrix=adj_matrix,
    )

    # Convert the JAX boolean mask to a numpy array for easy indexing
    action_mask = np.asarray(padding_masks["mask_2x"])

    # Match training's action clipping behavior.
    action_space = getattr(raw_env, "action_space", None)
    action_low = (
        None if action_space is None else np.asarray(action_space.low, dtype=np.float32).ravel()
    )
    action_high = (
        None if action_space is None else np.asarray(action_space.high, dtype=np.float32).ravel()
    )

    # 8. Run simulation
    headless = bool(sim_cfg.headless)
    max_steps = sim_cfg.max_steps

    if sim_cfg.record_video:
        if max_steps is None:
            raise ValueError("simulation.max_steps is required when simulation.record_video=true")

        max_steps_i = int(max_steps)
        if max_steps_i <= 0:
            raise ValueError("simulation.max_steps must be > 0")

        if sim_cfg.video_output_path is None:
            eval_dir = create_evaluation_dir(model_path)
            output_path = eval_dir / "simulation.mp4"
        else:
            output_path = Path(hydra.utils.to_absolute_path(sim_cfg.video_output_path))
            eval_dir = output_path.parent
            eval_dir.mkdir(parents=True, exist_ok=True)

        result = record_episode(
            env=env,
            policy=policy,
            seed=seed,
            max_steps=max_steps_i,
            action_low=action_low,
            action_high=action_high,
            action_mask=action_mask,
            output_path=output_path,
            camera_id=sim_cfg.camera_id,
        )

        save_evaluation_metadata(
            eval_dir=eval_dir,
            morphology_override_path=sim_cfg.morphology_override,
            seed=seed,
            max_steps=max_steps_i,
            result=result,
        )
        final_dist_str = "n/a" if result.final_xy_dist is None else f"{result.final_xy_dist:.3f}"
        print(f"Video saved to {output_path}")
        print(
            "episode done: "
            f"return={result.return_:.6f}, len={result.length}, "
            f"target_reached={result.reached_target}, final_xy_dist={final_dist_str}"
        )
    elif headless:
        if max_steps is None:
            raise ValueError("simulation.max_steps is required when simulation.headless=true")

        max_steps_i = int(max_steps)
        if max_steps_i <= 0:
            raise ValueError("simulation.max_steps must be > 0")

        result = rollout_headless(
            env=env,
            policy=policy,
            seed=seed,
            max_steps=max_steps_i,
            action_low=action_low,
            action_high=action_high,
            action_mask=action_mask,
        )
        final_dist_str = "n/a" if result.final_xy_dist is None else f"{result.final_xy_dist:.3f}"
        print(
            "episode done: "
            f"return={result.return_:.6f}, len={result.length}, "
            f"target_reached={result.reached_target}, final_xy_dist={final_dist_str}"
        )
    else:
        max_steps_val = None
        if max_steps is not None:
            max_steps_i = int(max_steps)
            if max_steps_i <= 0:
                raise ValueError("simulation.max_steps must be > 0")
            max_steps_val = max_steps_i

        model_dt = float(state0.mj_model.opt.timestep)
        control_dt = model_dt * float(training.environment.num_physics_steps_per_control_step)

        rollout_viewer(
            env=env,
            policy=policy,
            seed=seed,
            state=state0,
            control_dt=control_dt,
            max_steps=max_steps_val,
            action_low=action_low,
            action_high=action_high,
            action_mask=action_mask,
        )

    env.close()


if __name__ == "__main__":
    register_configs()
    main()
