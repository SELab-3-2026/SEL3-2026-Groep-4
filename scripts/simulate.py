"""Simulate a trained policy in the MuJoCo viewer.

Uses Hydra to load the same BrittleStarConfig that was used during training.
Override settings via CLI, e.g.:
    python scripts/simulate.py morphology=3_arms

To replay a run using the *exact* Hydra config used during training, pass:
    python scripts/simulate.py simulation.trained_config_path=runs/.../.hydra/config.yaml \
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
from omegaconf import DictConfig, OmegaConf, open_dict

from brittle_star_project import Backend, BrittleStarEnv, BrittleStarEnvFactory
from brittle_star_project.configs.main_config import BrittleStarConfig
from brittle_star_project.configs.register_configs import register_configs
from brittle_star_project.environment.padded_obs_wrapper import (
    compute_padding_masks,
    pad_observation,
)

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


def _dense_layer_sizes_from_params(params: Any) -> list[int]:
    """Infer GenericDenseLayersWithActivation.layer_sizes from a Flax params tree."""

    try:
        dense_params = params["params"]
    except Exception as exc:
        raise ValueError("Unexpected sensor params structure (missing 'params')") from exc

    layer_sizes: list[int] = []
    idx = 0
    while True:
        key = f"Dense_{idx}"
        if key not in dense_params:
            break
        kernel = dense_params[key]["kernel"]
        layer_sizes.append(int(np.asarray(kernel).shape[1]))
        idx += 1

    if not layer_sizes:
        raise ValueError("Could not infer Dense_* layers from sensor params")
    return layer_sizes


def _infer_action_dim_from_actor_params(params: Any) -> int | None:
    """Best-effort infer action_dim from a Flax Actor params tree."""

    try:
        dense0 = params["params"]["Dense_0"]
        bias = dense0.get("bias")
        kernel = dense0.get("kernel")
    except Exception:
        return None

    if bias is not None:
        try:
            return int(np.asarray(bias).shape[0])
        except Exception:
            return None

    if kernel is not None:
        try:
            return int(np.asarray(kernel).shape[1])
        except Exception:
            return None

    return None


def _has_cli_override(overrides: list[str], key: str) -> bool:
    prefixes = (f"{key}=", f"{key}.", f"+{key}=", f"+{key}.")
    return any(str(o).startswith(prefixes) for o in overrides)


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

        layer_sizes = _dense_layer_sizes_from_params(sensor_params)
        self._sensor = GenericDenseLayersWithActivation(layer_sizes=layer_sizes)
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

            # Accept a plain dict-shaped Flax params mapping commonly produced
            # by saving `agent_state.params` directly. Typical keys are
            # 'sensor_params' and 'actor_params', or sometimes nested under 'params'.
            if isinstance(restored_obj, dict):
                # Top-level params dict
                params_sub = restored_obj.get("params", {})
                sensor_params = restored_obj.get("sensor_params") or params_sub.get("sensor_params")
                actor_params = restored_obj.get("actor_params") or params_sub.get("actor_params")
                critic_params = restored_obj.get("critic_params") or params_sub.get("critic_params")
                feature_extractor_params = restored_obj.get(
                    "feature_extractor_params"
                ) or params_sub.get("feature_extractor_params")
                # Some checkpoints only save actor+sensor as top-level
                if sensor_params is not None and actor_params is not None:
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

        ckpt_action_dim = _infer_action_dim_from_actor_params(actor_params)
        if ckpt_action_dim is not None and ckpt_action_dim != action_dim:
            raise ValueError(
                "Checkpoint/env mismatch: "
                f"checkpoint expects action_dim={ckpt_action_dim}, "
                f"env provides action_dim={action_dim}. "
                "Use the same Hydra config (morphology/arena/environment) "
                "that was used during training."
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
    return bool(getattr(state, "terminated", False) or getattr(state, "truncated", False))


def _rollout_one_episode_headless(
    *,
    env: BrittleStarEnv,
    policy: CleanRLPPOPolicy,
    seed: int,
    max_steps: int,
    action_low: np.ndarray | None,
    action_high: np.ndarray | None,
    padding_masks: dict[str, Any] | None,
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
        obs_dict = observations or {}
        if padding_masks is not None:
            obs_dict = pad_observation(obs_dict, padding_masks)

        action = policy.act(observations=obs_dict)
        action = _maybe_clip_action(action, action_low, action_high)

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
    action_low: np.ndarray | None,
    action_high: np.ndarray | None,
    padding_masks: dict[str, Any] | None,
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

            obs_dict = observations or {}
            if padding_masks is not None:
                obs_dict = pad_observation(obs_dict, padding_masks)

            action = policy.act(observations=obs_dict)
            action = _maybe_clip_action(action, action_low, action_high)
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


def _load_trained_config(path: Path) -> DictConfig:
    """Load a trained config YAML.

    Supports both:
    - Hydra's run config (e.g. runs/.../.hydra/config.yaml)
    - This project's logger metadata YAMLs, which may contain
      ``!!python/object/apply:...`` tags for Enums.

    For safety, we *do not* execute Python constructors from YAML; we only
    treat these tags as data and extract their scalar arguments.
    """

    if not path.exists():
        raise FileNotFoundError(f"trained_config_path does not exist: '{path}'.")
    if not path.is_file():
        raise ValueError(f"trained_config_path must be a file, got: '{path}'.")

    try:
        return OmegaConf.load(path)
    except Exception as exc:
        python_apply_prefix = "tag:yaml.org,2002:python/object/apply:"

        class _SafeLoaderWithPythonApply(yaml.SafeLoader):
            pass

        def _construct_python_apply(
            loader: yaml.SafeLoader,
            _tag_suffix: str,
            node: yaml.Node,
        ) -> Any:
            if isinstance(node, yaml.SequenceNode):
                seq = loader.construct_sequence(node)
                if len(seq) == 1:
                    return seq[0]
                return seq
            if isinstance(node, yaml.MappingNode):
                return loader.construct_mapping(node)
            return loader.construct_scalar(node)

        _SafeLoaderWithPythonApply.add_multi_constructor(
            python_apply_prefix, _construct_python_apply
        )

        try:
            data = yaml.load(path.read_text(encoding="utf-8"), Loader=_SafeLoaderWithPythonApply)
        except Exception as yaml_exc:
            raise ValueError(
                "Failed to load trained_config_path as YAML. "
                "If this is a Hydra run, pass the run's '.hydra/config.yaml' file. "
                f"Got: '{path}'."
            ) from yaml_exc

        if not isinstance(data, dict):
            raise ValueError(
                "trained_config_path must contain a YAML mapping (dict-like) at the root. "
                f"Got type={type(data).__name__} from '{path}'."
            ) from exc

        # Normalize known enum-like strings to their Enum *names* so OmegaConf's
        # structured config merge behaves like the normal Hydra config.
        from brittle_star_project.environment.env_types import Task

        env_cfg = data.get("environment")
        if isinstance(env_cfg, dict) and isinstance(env_cfg.get("task"), str):
            task_str = str(env_cfg["task"])
            try:
                env_cfg["task"] = Task[task_str].name
            except Exception:
                try:
                    env_cfg["task"] = Task(task_str).name
                except Exception:
                    pass

        return OmegaConf.create(data)


@hydra.main(config_path="../configs", config_name="main_config", version_base="1.3")
def main(dict_cfg: DictConfig) -> None:
    # Compose against the structured schema first, so missing keys are validated.
    cfg = OmegaConf.merge(OmegaConf.structured(BrittleStarConfig), dict_cfg)

    # Optional: override env-defining sections (morphology/arena/environment/architecture)
    # using the exact Hydra config that was used for training.
    trained_cfg_path = cfg.simulation.trained_config_path
    if trained_cfg_path:
        overrides_raw = OmegaConf.select(cfg, "hydra.overrides.task") or []
        overrides = [str(o) for o in overrides_raw]

        trained_cfg_path_abs = Path(hydra.utils.to_absolute_path(trained_cfg_path))
        trained_cfg = _load_trained_config(trained_cfg_path_abs)
        if "hydra" in trained_cfg:
            with open_dict(trained_cfg):
                del trained_cfg["hydra"]

        with open_dict(cfg):
            for key in ("morphology", "arena", "environment", "architecture"):
                if key in trained_cfg and not _has_cli_override(overrides, key):
                    base_node = OmegaConf.select(cfg, key)
                    override_node = OmegaConf.select(trained_cfg, key)
                    try:
                        cfg[key] = OmegaConf.merge(base_node, override_node)
                    except Exception as exc:
                        raise ValueError(
                            "Failed to merge trained config into the active Hydra config. "
                            f"Key={key!r}, trained_config_path='{trained_cfg_path_abs}'."
                        ) from exc

    # Convert DictConfig to structured dataclass.
    config: BrittleStarConfig = OmegaConf.to_object(cfg)

    backend = Backend.MJC
    seed = int(config.experiment.seed)

    if getattr(config.architecture, "name", None) != "centralized":
        raise ValueError(
            "simulate.py currently only supports architecture=centralized. "
            f"Got architecture.name={getattr(config.architecture, 'name', None)!r}. "
            "(Training supports decentralized, but simulation wiring for it isn't implemented.)"
        )

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

    # Match training's padded observation layout for amputated morphologies.
    padding_masks = compute_padding_masks(config.morphology.segments_per_arm)

    # Match training's action clipping behavior.
    action_space = getattr(raw_env, "action_space", None)
    action_low = (
        None if action_space is None else np.asarray(action_space.low, dtype=np.float32).ravel()
    )
    action_high = (
        None if action_space is None else np.asarray(action_space.high, dtype=np.float32).ravel()
    )

    # ======= MODEL SETUP =======
    nu = int(state0.mj_model.nu)
    policy = CleanRLPPOPolicy.load(model_path, action_dim=nu)

    # Helpful early failure when configs don't match the checkpoint.
    observations0 = _get_observations(state0)
    obs0_dict = pad_observation(observations0 or {}, padding_masks)
    env_obs_dim = int(_transform_obs_dict(obs0_dict).shape[0])
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
            action_low=action_low,
            action_high=action_high,
            padding_masks=padding_masks,
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
            action_low=action_low,
            action_high=action_high,
            padding_masks=padding_masks,
        )

    env.close()


if __name__ == "__main__":
    register_configs()
    main()
