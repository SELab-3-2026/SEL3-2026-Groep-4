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
from omegaconf import DictConfig, OmegaConf


from brittle_star_project.configs.main_config import BrittleStarConfig
from brittle_star_project.configs.register_configs import register_configs

from brittle_star_project.evaluation.checkpoint import load_metadata, metadata_to_configs
from brittle_star_project.evaluation.eval_env_builder import build_eval_env
from brittle_star_project.evaluation.rollout import rollout_headless, rollout_viewer
from brittle_star_project.evaluation.video import (
    record_episode,
    create_evaluation_dir,
    save_evaluation_metadata,
)


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

    seed = int(cfg.experiment.seed)

    # 4-7. Build evaluation environment and policy
    override_path = None
    if sim_cfg.morphology_override is not None:
        override_path = Path(hydra.utils.to_absolute_path(sim_cfg.morphology_override))

    bundle = build_eval_env(
        model_path=model_path,
        training=training,
        metadata=metadata,
        morphology_override_path=override_path,
    )

    env = bundle.env
    policy = bundle.policy
    action_low = bundle.action_low
    action_high = bundle.action_high
    action_mask = bundle.action_mask

    state0 = env.reset(seed=seed)

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
