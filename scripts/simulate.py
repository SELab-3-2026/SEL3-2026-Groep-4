"""Simulate a trained policy in the MuJoCo viewer.

Automatically extracts the training configuration (morphology, environment, etc.)
from the sidecar metadata YAML file to ensure simulation perfectly matches training.
Override simulation settings via CLI, e.g.:
    uv run scripts/simulate.py \
        simulation.morphology_override=config/morphology/3_arms.yaml \
        simulation.model_path=runs/.../final_model.flax
"""

from __future__ import annotations

import itertools
import time
from pathlib import Path
from typing import Any

import flax
import hydra
import jax
import jax.numpy as jnp
import numpy as np
import yaml
from omegaconf import DictConfig, OmegaConf

from brittle_star_project import Backend, BrittleStarEnv, BrittleStarEnvFactory
from brittle_star_project.configs.main_config import BrittleStarConfig
from brittle_star_project.configs.register_configs import register_configs
from brittle_star_project.environment.padded_obs_wrapper import compute_padding_masks
from brittle_star_project.environment.obs_processing import create_obs_processor
from brittle_star_project.environment.env_config import (
    MorphologyConfig,
    ArenaConfig,
    EnvConfig,
    ObservationBoundsConfig,
)


class PolicyAgent:
    """Wraps a trained Flax actor for deterministic inference."""

    def __init__(
        self,
        *,
        sensor_params: Any,
        actor_params: Any,
        action_dim: int,
        obs_processor: Any,
    ) -> None:
        from brittle_star_project.MLPs.mlps import Actor, GenericDenseLayersWithActivation

        # Infer layer sizes from params
        try:
            dense_params = (
                sensor_params.get("params", {})
                if isinstance(sensor_params, dict)
                else sensor_params["params"]
            )
        except Exception:
            dense_params = sensor_params

        layer_sizes = []
        idx = 0
        while True:
            key = f"Dense_{idx}"
            if key not in dense_params:
                break
            layer_sizes.append(int(np.asarray(dense_params[key]["kernel"]).shape[1]))
            idx += 1

        if not layer_sizes:
            raise ValueError("Could not infer Dense_* layers from sensor params")

        self._sensor = GenericDenseLayersWithActivation(layer_sizes=layer_sizes)
        self._actor = Actor(action_dim=action_dim)
        self._sensor_apply = jax.jit(self._sensor.apply)
        self._actor_apply = jax.jit(self._actor.apply)
        self._params = {
            "sensor_params": sensor_params,
            "actor_params": actor_params,
        }
        self._obs_processor = obs_processor

    @staticmethod
    def load(
        path: Path,
        *,
        action_dim: int,
        obs_processor: Any,
    ) -> "PolicyAgent":
        payload = path.read_bytes()
        restored = flax.serialization.msgpack_restore(payload)

        sensor_params = None
        actor_params = None

        # Extract params from restored checkpoint
        if isinstance(restored, dict):
            params_sub = restored.get("params", {})
            sensor_params = restored.get("sensor_params") or params_sub.get("sensor_params")
            actor_params = restored.get("actor_params") or params_sub.get("actor_params")
        elif isinstance(restored, (list, tuple)) and len(restored) >= 2:
            params_part = restored[1]
            if isinstance(params_part, dict):
                sensor_params = params_part.get("0", params_part.get(0))
                actor_params = params_part.get("1", params_part.get(1))
            elif isinstance(params_part, (list, tuple)) and len(params_part) >= 2:
                sensor_params = params_part[0]
                actor_params = params_part[1]

        if sensor_params is None or actor_params is None:
            raise ValueError(f"Could not extract sensor and actor params from checkpoint: {path}")

        return PolicyAgent(
            sensor_params=sensor_params,
            actor_params=actor_params,
            action_dim=action_dim,
            obs_processor=obs_processor,
        )

    def act(self, *, observations: dict[str, Any]) -> np.ndarray:
        batched_obs = jax.tree.map(lambda x: jnp.asarray(x)[None, ...], observations)
        obs = self._obs_processor(batched_obs)[0]
        hidden = self._sensor_apply(self._params["sensor_params"], obs)
        mean, _log_std = self._actor_apply(self._params["actor_params"], hidden)

        # Always evaluate with the actor mean.
        # (Sampling adds exploration noise, which is useful for training but not for evaluation.)
        return np.asarray(mean, dtype=np.float32).ravel()


def _get_observations(state: Any) -> dict[str, Any] | None:
    return getattr(state, "observations", None)


def _get_xy_distance_to_target(observations: dict[str, Any]) -> float | None:
    return float(np.asarray(observations["xy_distance_to_target"]).reshape(-1)[0])


def _target_reached(*, state: Any) -> bool:
    return bool(getattr(state, "terminated", False) or getattr(state, "truncated", False))


def _maybe_clip_action(
    action: np.ndarray,
    low: np.ndarray | None,
    high: np.ndarray | None,
) -> np.ndarray:
    if low is None or high is None:
        return action
    low = np.asarray(low, dtype=np.float32).ravel()
    high = np.asarray(high, dtype=np.float32).ravel()
    if low.shape != action.shape or high.shape != action.shape:
        return action
    return np.clip(action, low, high)


def _rollout_headless(
    *,
    env: BrittleStarEnv,
    policy: PolicyAgent,
    seed: int,
    max_steps: int,
    action_low: np.ndarray | None,
    action_high: np.ndarray | None,
    action_mask: np.ndarray | None = None,
) -> tuple[float, int, bool, float | None]:
    state = env.reset(seed=seed)

    ep_return = 0.0
    observations = _get_observations(state)
    prev_dist = _get_xy_distance_to_target(observations)
    reached_target = _target_reached(state=state)

    steps = 0
    for _ in range(int(max_steps)):
        obs_dict = observations or {}

        action = policy.act(observations=obs_dict)
        if action_mask is not None:
            action = action[action_mask]
        action = _maybe_clip_action(action, action_low, action_high)

        state = env.step(state=state, action=action)
        steps += 1

        observations = _get_observations(state)
        cur_dist = _get_xy_distance_to_target(observations)
        if prev_dist is not None and cur_dist is not None:
            ep_return += prev_dist - cur_dist
        prev_dist = cur_dist

        reached_target = _target_reached(state=state)
        if reached_target:
            break

    final_dist = _get_xy_distance_to_target(observations)
    return ep_return, steps, reached_target, final_dist


def _rollout_viewer(
    *,
    env: BrittleStarEnv,
    policy: PolicyAgent,
    seed: int,
    state: Any,
    control_dt: float,
    max_steps: int | None,
    action_low: np.ndarray | None,
    action_high: np.ndarray | None,
    action_mask: np.ndarray | None = None,
) -> None:
    import mujoco.viewer

    model = state.mj_model
    data = state.mj_data

    episode_return = 0.0
    observations = _get_observations(state)
    prev_dist = _get_xy_distance_to_target(observations)
    reached_target = _target_reached(state=state)

    steps = 0
    # Use the viewer as a context manager to avoid GLX teardown races.
    with mujoco.viewer.launch_passive(model, data) as viewer:
        step_iter = range(int(max_steps)) if max_steps is not None else itertools.count()
        for _step_idx in step_iter:
            if not viewer.is_running():
                break
            step_start = time.time()

            obs_dict = observations or {}

            action = policy.act(observations=obs_dict)
            if action_mask is not None:
                action = action[action_mask]
            action = _maybe_clip_action(action, action_low, action_high)

            # The passive viewer runs a GUI thread; protect MuJoCo state mutation.
            with viewer.lock():
                state = env.step(state=state, action=action)

            if not viewer.is_running():
                break
            viewer.sync()

            steps += 1

            observations = _get_observations(state)
            cur_dist = _get_xy_distance_to_target(observations)
            if prev_dist is not None and cur_dist is not None:
                episode_return += prev_dist - cur_dist
            prev_dist = cur_dist

            reached_target = _target_reached(state=state)
            if reached_target:
                break

            remaining = control_dt - (time.time() - step_start)
            if remaining > 0:
                time.sleep(remaining)

    dist = _get_xy_distance_to_target(observations)
    dist_str = "n/a" if dist is None else f"{dist:.3f}"
    print(
        "episode done: "
        f"return={episode_return:.6f}, len={steps}, "
        f"target_reached={reached_target}, final_xy_dist={dist_str}"
    )


def _load_metadata_yaml(model_path: Path) -> dict:
    """Discover and load the sidecar metadata YAML file."""
    metadata_path = model_path.with_name(model_path.stem + "_metadata.yaml")
    if not metadata_path.exists():
        raise FileNotFoundError(
            f"Could not find metadata YAML for {model_path.name}. Expected it at {metadata_path}"
        )
    with open(metadata_path, "r") as f:
        return yaml.safe_load(f)


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
    metadata = _load_metadata_yaml(model_path)

    # 3. Reconstruct typed configs from metadata
    trained_morphology = OmegaConf.to_object(
        OmegaConf.merge(OmegaConf.structured(MorphologyConfig), metadata.get("morphology", {}))
    )
    trained_arena = OmegaConf.to_object(
        OmegaConf.merge(OmegaConf.structured(ArenaConfig), metadata.get("arena", {}))
    )

    env_dict = metadata.get("environment", {})
    if isinstance(env_dict.get("task"), str):
        from brittle_star_project.environment.env_types import Task

        try:
            env_dict["task"] = Task[env_dict["task"]].name
        except Exception:
            try:
                env_dict["task"] = Task(env_dict["task"]).name
            except Exception:
                pass

    trained_environment = OmegaConf.to_object(
        OmegaConf.merge(OmegaConf.structured(EnvConfig), env_dict)
    )
    trained_obs_bounds = OmegaConf.to_object(
        OmegaConf.merge(
            OmegaConf.structured(ObservationBoundsConfig), metadata.get("obs_bounds", {})
        )
    )

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
        env_morphology = trained_morphology

    # 5. Build obs_processor with TRAINING morphology padding masks always
    padding_masks = compute_padding_masks(
        segments_per_arm=env_morphology.segments_per_arm,
    )
    obs_processor = create_obs_processor(
        bounds_dict=trained_obs_bounds.to_bounds_dict(),
        padding_masks=padding_masks,
    )

    # 6. Build environment
    backend = Backend.MJC
    seed = int(cfg.experiment.seed)

    factory = BrittleStarEnvFactory()
    raw_env = factory.create_environment(
        backend,
        env_morphology,
        trained_arena,
        trained_environment,
    )
    env = BrittleStarEnv(
        raw_env,
        backend=backend,
        config=trained_environment,
        morphology_config=env_morphology,
    )

    state0 = env.reset(seed=seed)

    # Calculate the action dimension the model was trained with
    trained_action_dim = sum(trained_morphology.segments_per_arm) * 2

    # 7. Load policy
    policy = PolicyAgent.load(
        model_path, action_dim=trained_action_dim, obs_processor=obs_processor
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

    if headless:
        if max_steps is None:
            raise ValueError("simulation.max_steps is required when simulation.headless=true")

        max_steps_i = int(max_steps)
        if max_steps_i <= 0:
            raise ValueError("simulation.max_steps must be > 0")

        ep_return, ep_len, reached_target, final_dist = _rollout_headless(
            env=env,
            policy=policy,
            seed=seed,
            max_steps=max_steps_i,
            action_low=action_low,
            action_high=action_high,
            action_mask=action_mask,
        )
        final_dist_str = "n/a" if final_dist is None else f"{final_dist:.3f}"
        print(
            "episode done: "
            f"return={ep_return:.6f}, len={ep_len}, "
            f"target_reached={reached_target}, final_xy_dist={final_dist_str}"
        )
    else:
        max_steps_val = None
        if max_steps is not None:
            max_steps_i = int(max_steps)
            if max_steps_i <= 0:
                raise ValueError("simulation.max_steps must be > 0")
            max_steps_val = max_steps_i

        model_dt = float(state0.mj_model.opt.timestep)
        control_dt = model_dt * float(trained_environment.num_physics_steps_per_control_step)

        _rollout_viewer(
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
