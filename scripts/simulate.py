"""Simulate a trained policy in the MuJoCo viewer.

Uses Hydra to load the same BrittleStarConfig that was used during training.
Override settings via CLI, e.g.:
    python scripts/simulate.py morphology=3_arms
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
from omegaconf import DictConfig, OmegaConf

from brittle_star_project import BrittleStarEnv, BrittleStarEnvFactory
from brittle_star_project.configs.main_config import BrittleStarConfig
from brittle_star_project.configs.register_configs import register_configs

_ALLOWED_OBS_KEYS = {
    "joint_position",
    "joint_velocity",
    "joint_actuator_force",
    "actuator_force",
    "disk_position",
    "disk_rotation",
    "disk_linear_velocity",
    "disk_angular_velocity",
    "unit_xy_direction_to_target",
    "xy_distance_to_target",
}

def _transform_obs_dict(obs_dict: dict[str, Any]) -> jnp.ndarray:
    """Flatten the env's observation dict into a 1D vector.

    Matches training behavior:
    - only includes keys in _ALLOWED_OBS_KEYS
    - iterates keys in sorted order for stable layout
    - skips empty arrays
    """
    parts: list[jnp.ndarray] = []
    for key in sorted(obs_dict.keys()):
        if key not in _ALLOWED_OBS_KEYS:
            continue
        arr = jnp.asarray(obs_dict[key])
        if arr.size == 0:
            continue
        parts.append(arr.reshape((-1,)))

    if not parts:
        return jnp.zeros((0,), dtype=jnp.float32)
    return jnp.concatenate(parts, axis=0)


# A minimal policy class to load a CleanRL/Flax checkpoint and run inference.
class CleanRLPPOPolicy:
    def __init__(
        self,
        *,
        sensor_params: Any,
        actor_params: Any,
        action_dim: int,
    ) -> None:
        from brittle_star_project.MLPs.mlps import Actor, GenericDenseLayersWithActivation

        hidden_dim = int(sensor_params["params"]["Dense_0"]["kernel"].shape[1])

        self._sensor = GenericDenseLayersWithActivation(layer_sizes=[hidden_dim, hidden_dim])
        self._actor = Actor(action_dim=action_dim)
        self._sensor_apply = jax.jit(self._sensor.apply)
        self._actor_apply = jax.jit(self._actor.apply)
        self._params = {
            "sensor_params": sensor_params,
            "actor_params": actor_params,
        }

    @staticmethod
    def load(
        path: Path,
        *,
        action_dim: int,
    ) -> "CleanRLPPOPolicy":
        def _get_index(container: Any, idx: int) -> Any:
            if isinstance(container, (list, tuple)):
                return container[idx]
            if isinstance(container, dict):
                return container.get(idx, container.get(str(idx)))
            raise KeyError(idx)

        def _looks_like_indexed_dict(container: Any) -> bool:
            return (
                isinstance(container, dict)
                and container
                and all(str(k).isdigit() for k in container.keys())
            )

        def _parse_checkpoint(restored_obj: Any) -> tuple[Any, Any, Any, Any, Any]:
            """Extract checkpoint parts.

            Returns (config_dict, sensor_params, actor_params, critic_params,
            feature_extractor_params).

            PPOTrainer saves:
              flax.serialization.to_bytes([
                config_dict,
                [sensor_params, actor_params, critic_params, feature_extractor_params],
              ])

            msgpack_restore() may restore lists as dicts keyed by string indices
            ("0", "1", ...), so we accept both shapes.
            """

            cfg_part: Any | None = None
            params_part: Any = restored_obj

            if isinstance(restored_obj, (list, tuple)) and len(restored_obj) >= 2:
                cfg_part = restored_obj[0]
                params_part = restored_obj[1]
            elif _looks_like_indexed_dict(restored_obj) and (
                "0" in restored_obj or "1" in restored_obj
            ):
                cfg_part = restored_obj.get("0", restored_obj.get(0))
                params_part = restored_obj.get("1", restored_obj.get(1))

            if _looks_like_indexed_dict(params_part):
                sensor_params = _get_index(params_part, 0)
                actor_params = _get_index(params_part, 1)
                critic_params = _get_index(params_part, 2)
                feature_extractor_params = _get_index(params_part, 3)
                if sensor_params is None or actor_params is None:
                    raise ValueError("Missing required params in checkpoint")
                return (
                    cfg_part,
                    sensor_params,
                    actor_params,
                    critic_params,
                    feature_extractor_params,
                )

            if isinstance(params_part, (list, tuple)) and len(params_part) >= 2:
                sensor_params = params_part[0]
                actor_params = params_part[1]
                critic_params = params_part[2] if len(params_part) >= 3 else None
                feature_extractor_params = params_part[3] if len(params_part) >= 4 else None
                return (
                    cfg_part,
                    sensor_params,
                    actor_params,
                    critic_params,
                    feature_extractor_params,
                )

            raise ValueError(
                f"Unexpected checkpoint structure in {path}. "
                "Expected [config_dict, [sensor_params, actor_params, critic_params, "
                "feature_extractor_params]] or an equivalent dict-indexed variant."
            )

        payload = path.read_bytes()
        restored = flax.serialization.msgpack_restore(payload)
        _cfg_dict, sensor_params, actor_params, _critic_params, _feature_extractor_params = (
            _parse_checkpoint(restored)
        )

        return CleanRLPPOPolicy(
            sensor_params=sensor_params,
            actor_params=actor_params,
            action_dim=action_dim,
        )

    def act(self, *, observations: dict[str, Any]) -> np.ndarray:
        obs = _transform_obs_dict(observations)
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
    return bool(getattr(state, "terminated", False))


def _rollout_one_episode_headless(
    *,
    env: BrittleStarEnv,
    policy: CleanRLPPOPolicy,
    seed: int,
    max_steps: int,
) -> tuple[float, int, bool, float | None]:
    """Run one rollout up to max_steps.

    Returns (return, length, reached_target, final_xy_dist).

    Note: In the MJC backend, the raw env reward can be 0.0; we compute a simple
    progress reward based on xy_distance_to_target.
    """

    state = env.reset(seed=seed)

    ep_return = 0.0
    observations = _get_observations(state)
    prev_dist = _get_xy_distance_to_target(observations)
    reached_target = _target_reached(state=state)

    steps = 0
    for _ in range(int(max_steps)):
        action = policy.act(observations=observations)

        nu = int(state.mj_model.nu)
        if nu > 0 and action.shape != (nu,):
            raise ValueError(f"Policy returned action shape {action.shape}, expected ({nu},)")

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


def _run_one_episode_viewer(
    *,
    env: BrittleStarEnv,
    policy: CleanRLPPOPolicy,
    seed: int,
    state: Any,
    control_dt: float,
    max_steps: int | None,
) -> None:
    import mujoco.viewer

    model = state.mj_model
    data = state.mj_data

    _ = int(seed)
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

            action = policy.act(observations=observations or {})
            if model.nu > 0 and action.shape != (int(model.nu),):
                raise ValueError(
                    f"Policy returned action shape {action.shape}, expected ({int(model.nu)},)"
                )

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


def _infer_checkpoint_obs_dim(policy: CleanRLPPOPolicy) -> int | None:
    """Best-effort read of the first Dense kernel input dim (obs dim)."""

    try:
        kernel = policy._params["sensor_params"]["params"]["Dense_0"]["kernel"]
        return int(getattr(kernel, "shape")[0])
    except Exception:
        return None


@hydra.main(config_path="../configs", config_name="main_config", version_base="1.3")
def main(dict_cfg: DictConfig) -> None:
    # Convert DictConfig to structured dataclass, ensuring the root schema is applied.
    config: BrittleStarConfig = OmegaConf.to_object(
        OmegaConf.merge(OmegaConf.structured(BrittleStarConfig), dict_cfg)
    )

    backend = config.simulation.backend
    seed = int(config.experiment.seed)

    model_path_str = config.simulation.model_path
    if model_path_str is None:
        raise ValueError(
            "simulation.model_path must be set to a .flax checkpoint (e.g. final_model.flax)"
        )

    # Hydra chdir changes CWD; resolve relative paths relative to the invocation.
    model_path = Path(hydra.utils.to_absolute_path(model_path_str))
    if model_path.suffix != ".flax":
        raise ValueError(f"Expected a '.flax' checkpoint, got '{model_path.name}'.")

    # ======= ENVIRONMENT SETUP =======
    factory = BrittleStarEnvFactory()
    raw_env = factory.create_environment(
        backend,
        config.morphology,
        config.arena,
        config.environment,
    )
    env = BrittleStarEnv(
        raw_env,
        backend=backend,
        config=config.environment,
        morphology_config=config.morphology,
    )

    state0 = env.reset(seed=seed)

    # ======= MODEL SETUP =======
    nu = int(state0.mj_model.nu)
    policy = CleanRLPPOPolicy.load(model_path, action_dim=nu)

    # Helpful early failure when configs don't match the checkpoint.
    observations0 = _get_observations(state0)
    env_obs_dim = int(_transform_obs_dict(observations0 or {}).shape[0])
    ckpt_obs_dim = _infer_checkpoint_obs_dim(policy)
    
    if ckpt_obs_dim is not None and ckpt_obs_dim != env_obs_dim:
        raise ValueError(
            "Checkpoint/env mismatch: "
            f"checkpoint expects obs_dim={ckpt_obs_dim}, env provides obs_dim={env_obs_dim}. "
            "Use the same Hydra config (morphology/arena/environment) "
            "that was used during training."
        )
    
    # ======= SIMULATION =======
    headless = bool(config.simulation.headless)
    max_steps = config.simulation.max_steps

    if headless:
        if max_steps is None:
            raise ValueError("simulation.max_steps is required when simulation.headless=true")
        max_steps_i = int(max_steps)
        if max_steps_i <= 0:
            raise ValueError("simulation.max_steps must be > 0")

        ep_return, ep_len, reached_target, final_dist = _rollout_one_episode_headless(
            env=env,
            policy=policy,
            seed=seed,
            max_steps=max_steps_i,
        )
        final_dist_str = "n/a" if final_dist is None else f"{final_dist:.3f}"
        print(
            "episode done: "
            f"return={ep_return:.6f}, len={ep_len}, "
            f"target_reached={reached_target}, final_xy_dist={final_dist_str}"
        )
    else:
        if max_steps is not None:
            max_steps_i = int(max_steps)
            if max_steps_i <= 0:
                raise ValueError("simulation.max_steps must be > 0")
            max_steps_val: int | None = max_steps_i
        else:
            max_steps_val = None

        model_dt = float(state0.mj_model.opt.timestep)
        control_dt = model_dt * float(config.environment.num_physics_steps_per_control_step)

        _run_one_episode_viewer(
            env=env,
            policy=policy,
            seed=seed,
            state=state0,
            control_dt=control_dt,
            max_steps=max_steps_val,
        )

    env.close()


if __name__ == "__main__":
    register_configs()
    main()
