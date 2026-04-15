from __future__ import annotations

import argparse
import itertools
import time
from pathlib import Path
from typing import Any
import flax
import jax
import jax.numpy as jnp
import numpy as np

from brittle_star_project import (
    Backend,
)
from brittle_star_project.environment import ArenaConfig, EnvConfig, MorphologyConfig, from_file

def _flatten_obs_dict(obs_dict: dict[str, Any]) -> jnp.ndarray:
    """Flatten the env's observation dict into a 1D vector.

    concatenates values in the dict's iteration order and skips empty arrays.
    """

    parts: list[jnp.ndarray] = []
    for v in obs_dict.values():
        arr = jnp.asarray(v)
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

        self._sensor = GenericDenseLayersWithActivation()
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

            Returns (args_dict, sensor_params, actor_params, critic_params,
            feature_extractor_params).

            `PPOTrainer` saves:
              flax.serialization.to_bytes(
                [vars(args), [sensor, actor, critic, feature_extractor]]
              )

            `msgpack_restore()` may restore lists as dicts keyed by string
            indices ("0", "1", ...), so we accept both shapes.
            """

            args_part: Any | None = None
            params_part: Any = restored_obj

            if isinstance(restored_obj, (list, tuple)) and len(restored_obj) >= 2:
                args_part = restored_obj[0]
                params_part = restored_obj[1]
            elif _looks_like_indexed_dict(restored_obj) and (
                "0" in restored_obj or "1" in restored_obj
            ):
                args_part = restored_obj.get("0", restored_obj.get(0))
                params_part = restored_obj.get("1", restored_obj.get(1))

            if _looks_like_indexed_dict(params_part):
                sensor_params = _get_index(params_part, 0)
                actor_params = _get_index(params_part, 1)
                critic_params = _get_index(params_part, 2)
                feature_extractor_params = _get_index(params_part, 3)
                if sensor_params is None or actor_params is None:
                    raise ValueError("Missing required params in checkpoint")
                return (
                    args_part,
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
                    args_part,
                    sensor_params,
                    actor_params,
                    critic_params,
                    feature_extractor_params,
                )

            raise ValueError(
                f"Unexpected checkpoint structure in {path}. "
                "Expected [args_dict, [sensor_params, actor_params, critic_params, "
                "feature_extractor_params]] "
                "or an equivalent dict-indexed variant."
            )

        payload = path.read_bytes()
        restored = flax.serialization.msgpack_restore(payload)
        _args_dict, sensor_params, actor_params, _critic_params, _feature_extractor_params = (
            _parse_checkpoint(restored)
        )

        return CleanRLPPOPolicy(
            sensor_params=sensor_params,
            actor_params=actor_params,
            action_dim=action_dim,
        )

    def act(self, *, observations: dict[str, Any]) -> np.ndarray:
        obs = _flatten_obs_dict(observations)
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
    env: Any,
    policy: CleanRLPPOPolicy,
    seed: int,
    max_steps: int,
) -> tuple[float, int, bool, float | None]:
    """Run one rollout up to `max_steps`.

    Returns (return, length, reached_target, final_xy_dist).
    """
    state = env.reset(seed=seed)

    ep_return = 0.0

    observations = _get_observations(state)
    prev_dist = _get_xy_distance_to_target(observations)
    reached_target = _target_reached(state=state)

    # NOTE: In the MJC backend, `state.reward` is always 0.0.
    # To get a meaningful return, we compute a simple progress reward:
    #   r_t = d_{t-1} - d_t
    # where d is `xy_distance_to_target`.
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
    env: Any,
    policy: CleanRLPPOPolicy,
    seed: int,
    state: Any,
    control_dt: float,
    max_steps: int | None,
) -> None:
    import mujoco.viewer

    model = state.mj_model
    data = state.mj_data

    seed = int(seed)
    episode_return = 0.0
    observations = _get_observations(state)
    prev_dist = _get_xy_distance_to_target(observations)
    reached_target = _target_reached(state=state)

    steps = 0
    # Use the viewer as a context manager to avoid GLX teardown races
    # (e.g. GLXBadDrawable from X_GLXSwapBuffers after a window is destroyed).
    with mujoco.viewer.launch_passive(model, data) as viewer:
        step_iter = (
            range(int(max_steps)) if max_steps is not None else itertools.count()
        )
        for _step_idx in step_iter:
            if not viewer.is_running():
                break
            step_start = time.time()

            action = policy.act(observations=observations)
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


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Run a trained policy for exactly one episode (viewer or headless)."
    )
    p.add_argument(
        "--config-path",
        type=str,
        default=None,
        help=(
            "Path to an environment JSON config (morphology/arena/env). "
            "If omitted, uses the environment defaults. "
            "Relative paths are resolved from the repository root."
        ),
    )
    p.add_argument(
        "--model",
        type=str,
        required=True,
        help=(
            "Path to the Flax checkpoint saved by scripts/train.py (final_model.flax)."
        ),
    )
    p.add_argument(
        "--headless",
        action="store_true",
        help="Run without the MuJoCo viewer (still exactly one episode).",
    )
    p.add_argument(
        "--max-steps",
        type=int,
        default=None,
        help=(
            "Number of control steps to run. "
            "In --headless mode this is required and acts as a fixed horizon. "
            "In viewer mode the default is infinite (run until window closed or target reached)."
        ),
    )
    p.add_argument(
        "--backend",
        choices=[b.value for b in Backend],
        default=Backend.MJC.value,
    )
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


def main() -> None:
    from brittle_star_project.environment import (
        BrittleStarEnv,
        BrittleStarEnvFactory,
    )

    args = parse_args()

    if args.config_path is None:
        morphology_cfg = MorphologyConfig()
        arena_cfg = ArenaConfig()
        env_cfg = EnvConfig()
    else:
        repo_root = Path(__file__).resolve().parents[1]
        config_path = Path(args.config_path)
        if not config_path.is_absolute():
            config_path = repo_root / config_path
        morphology_cfg, arena_cfg, env_cfg = from_file(str(config_path))

    # ======= ENVIRONMENT SETUP =======

    backend = Backend(args.backend)

    factory = BrittleStarEnvFactory()
    raw_env = factory.create_environment(backend, morphology_cfg, arena_cfg, env_cfg)
    env = BrittleStarEnv(
        raw_env,
        backend=backend,
        config=env_cfg,
        morphology_config=morphology_cfg,
    )

    seed_for_env = int(args.seed) if args.seed is not None else 0
    state = env.reset(seed=seed_for_env)

    # ======= MODEL SETUP =======

    # Extract the number of actuators (nu) from the environment's model, so we can pass it to the
    # policy/model.
    nu = int(state.mj_model.nu)

    model_path = Path(args.model)
    if model_path.name != "final_model.flax" or model_path.suffix != ".flax":
        raise ValueError(
            "Expected the training artifact 'final_model.flax', "
            f"got '{model_path.name}'."
        )

    policy = CleanRLPPOPolicy.load(
        model_path,
        action_dim=nu,
    )

    default_seed = seed_for_env

    # ======= SIMULATION =======

    if args.headless:
        if args.max_steps is None:
            raise ValueError("--max-steps is required in --headless mode")
        max_steps = int(args.max_steps)
        if max_steps <= 0:
            raise ValueError("--max-steps must be > 0")

        ep_seed = int(args.seed) if args.seed is not None else default_seed
        ep_return, ep_len, reached_target, final_dist = _rollout_one_episode_headless(
            env=env,
            policy=policy,
            seed=ep_seed,
            max_steps=max_steps,
        )
        final_dist_str = "n/a" if final_dist is None else f"{final_dist:.3f}"
        print(
            "episode done: "
            f"return={ep_return:.6f}, len={ep_len}, "
            f"target_reached={reached_target}, final_xy_dist={final_dist_str}"
        )
    else:
        max_steps: int | None
        if args.max_steps is None:
            max_steps = None
        else:
            max_steps = int(args.max_steps)
            if max_steps <= 0:
                raise ValueError("--max-steps must be > 0")

        model_dt = float(state.mj_model.opt.timestep)
        control_dt = model_dt * float(env_cfg.num_physics_steps_per_control_step)
        _run_one_episode_viewer(
            env=env,
            policy=policy,
            seed=int(args.seed) if args.seed is not None else default_seed,
            state=state,
            control_dt=control_dt,
            max_steps=max_steps,
        )

    env.close()


if __name__ == "__main__":
    main()
