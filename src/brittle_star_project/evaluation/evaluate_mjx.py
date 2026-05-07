"""MJX-based headless checkpoint evaluation.

This module provides a fast, JIT-compiled evaluation path using the MJX
(JAX-accelerated MuJoCo) backend. It is intended for evaluating checkpoints
*during* or *after* a training run, where the environment and policy are
already fully initialised.

The key functions are:

- `build_eval_rollout_fn` — builds and JIT-compiles a single-episode rollout function from the
  training environment and policy components.
- `evaluate_checkpoint_mjx` — runs that function for a given set of parameters and returns a typed
  `CheckpointEvalResult`.
- `append_checkpoint_eval_row` — persists the result to the run's
  ``metrics/checkpoint_evaluation.csv``, migrating old schemas automatically.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import jax
import jax.numpy as jnp


@dataclass
class CheckpointEvalResult:
    """Structured result from a single MJX checkpoint evaluation episode."""

    steps: int
    """Number of control steps taken (≤ max_steps)."""

    reached_target: bool
    """Whether the robot reached the target (terminated) before max_steps."""

    eval_return: float
    """Accumulated shaped reward over the episode."""

    final_xy_dist: float
    """XY distance to target at episode end. 0.0 when ``reached_target`` is True."""

    initial_xy_dist: float
    """XY distance to target at episode start."""


def build_eval_rollout_fn(
    *,
    env: Any,
    obs_processor: Callable,
    sensor_apply: Callable,
    actor_apply: Callable,
    action_low: jnp.ndarray,
    action_high: jnp.ndarray,
    reward_fn: Callable,
) -> Callable:
    """Build and JIT-compile a single-episode MJX evaluation rollout.

    The returned function has the signature::

        eval_fn(params: dict, seed: int, max_steps: int)
            -> (steps, reached_target, eval_return, final_xy_dist, initial_xy_dist)

    All outputs are JAX arrays. Convert to Python scalars before logging.

    Args:
        env: The training environment wrapper. Must expose ``env.raw`` with
            ``reset`` and ``step`` methods compatible with ``jax.vmap``.
        obs_processor: Observation normalisation / padding callable, as
            returned by ``create_obs_processor``.
        sensor_apply: The sensor network's ``apply`` method (JIT-compiled).
        actor_apply: The actor network's ``apply`` method (JIT-compiled).
        action_low: Per-joint action lower bound (JAX array, shape ``(action_dim,)``).
        action_high: Per-joint action upper bound (JAX array, shape ``(action_dim,)``).
        reward_fn: Shaped reward function with signature
            ``reward_fn(env_state, next_env_state) -> jnp.ndarray``.
            Typically the module-level ``reward_fn`` from ``PPOTrainer``.

    Returns:
        A JIT-compiled callable that runs one deterministic evaluation episode.
    """
    # vmap over a batch of 1 so the MJX API is satisfied without any
    # extra bookkeeping in the caller.
    reset_1 = jax.vmap(env.raw.reset)
    step_1 = jax.vmap(env.raw.step)

    def _eval_rollout(params: dict, seed: int, max_steps: int):
        rng = jax.random.PRNGKey(seed)
        rngs = jnp.asarray(jax.random.split(rng, 1))
        state = reset_1(rng=rngs)

        initial_xy_dist = jnp.squeeze(state.observations["xy_distance_to_target"])

        t0 = jnp.asarray(0, dtype=jnp.int32)
        done0 = jnp.squeeze(state.terminated | state.truncated)
        return0 = jnp.asarray(0.0, dtype=jnp.float32)

        def cond(carry):
            t, _state, done, _return_ = carry
            return jnp.logical_and(t < max_steps, jnp.logical_not(done))

        def body(carry):
            t, state, _done, return_ = carry

            obs = obs_processor(state.observations)
            hidden = sensor_apply(params["sensor_params"], obs)
            mean, _log_std = actor_apply(params["actor_params"], hidden)

            # Deterministic action: use the actor mean, no exploration noise.
            action = jnp.clip(mean, action_low, action_high)
            next_state = step_1(state=state, action=action)

            shaped_reward = reward_fn(state, next_state)
            return_ = return_ + jnp.squeeze(shaped_reward)

            done_next = jnp.squeeze(next_state.terminated | next_state.truncated)
            return (t + 1, next_state, done_next, return_)

        t, final_state, _done, return_ = jax.lax.while_loop(cond, body, (t0, state, done0, return0))

        reached_target = jnp.squeeze(final_state.terminated)
        final_xy_dist_raw = jnp.squeeze(final_state.observations["xy_distance_to_target"])
        # Clamp to 0 when the target was reached so downstream consumers
        # don't have to special-case "terminated" themselves.
        final_xy_dist = jnp.where(reached_target, 0.0, final_xy_dist_raw)

        return t, reached_target, return_, final_xy_dist, initial_xy_dist

    return jax.jit(_eval_rollout)


def evaluate_checkpoint_mjx(
    eval_fn: Callable,
    params: dict,
    *,
    seed: int,
    max_steps: int,
) -> CheckpointEvalResult:
    """Run one deterministic evaluation episode and return typed metrics.

    Args:
        eval_fn: A JIT-compiled function as returned by :func:`build_eval_rollout_fn`.
        params: Agent parameter dict (e.g. ``agent_state.params``).
        seed: Random seed for environment reset (controls target placement).
        max_steps: Maximum number of control steps before the episode is cut off.

    Returns:
        A :class:`CheckpointEvalResult` with all JAX arrays converted to
        plain Python scalars.
    """
    steps, reached, eval_return, final_xy_dist, initial_xy_dist = eval_fn(params, seed, max_steps)
    return CheckpointEvalResult(
        steps=int(steps),
        reached_target=bool(reached),
        eval_return=float(eval_return),
        final_xy_dist=float(final_xy_dist),
        initial_xy_dist=float(initial_xy_dist),
    )


_FIELDNAMES = [
    "checkpoint",
    "trained_timesteps",
    "eval_steps",
    "eval_return",
    "final_xy_dist",
    "initial_xy_dist",
    "reached_target",
]


def _migrate_csv_if_needed(csv_path: Path) -> None:
    """Rewrite the CSV with the canonical field names if the schema changed.

    Best-effort: any exception is silently swallowed so that a schema mismatch
    never causes a training crash.
    """
    try:
        with open(csv_path, "r", newline="") as f:
            header = next(csv.reader(f), None)

        if header is None or list(header) == _FIELDNAMES:
            return  # Nothing to migrate.

        migrated_rows: list[dict[str, Any]] = []
        with open(csv_path, "r", newline="") as f:
            for row in csv.DictReader(f):
                migrated_rows.append(
                    {
                        "checkpoint": row.get("checkpoint", row.get("iteration")),
                        "trained_timesteps": row.get("trained_timesteps"),
                        "eval_steps": row.get("eval_steps", row.get("steps_to_target")),
                        "eval_return": row.get("eval_return"),
                        "final_xy_dist": row.get("final_xy_dist"),
                        "initial_xy_dist": row.get("initial_xy_dist"),
                        "reached_target": row.get("reached_target"),
                    }
                )

        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=_FIELDNAMES)
            writer.writeheader()
            writer.writerows(migrated_rows)
    except Exception:
        pass  # Never crash training on a migration issue.


def append_checkpoint_eval_row(
    run_dir: str | Path,
    *,
    iteration: int,
    trained_timesteps: int,
    result: CheckpointEvalResult,
) -> Path:
    """Append one evaluation row to ``<run_dir>/metrics/checkpoint_evaluation.csv``.

    Creates the file (including the ``metrics/`` directory) if it does not yet
    exist. Migrates the file to the current schema if the header has changed.

    Args:
        run_dir: Root directory of the training run (Hydra's output dir).
        iteration: Training iteration number, used as the checkpoint identifier.
        trained_timesteps: Total environment steps taken at this checkpoint.
        result: Evaluation result as returned by :func:`evaluate_checkpoint_mjx`.

    Returns:
        Absolute path to the CSV file (useful for W&B sync).
    """
    metrics_dir = Path(run_dir) / "metrics"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    csv_path = metrics_dir / "checkpoint_evaluation.csv"

    if csv_path.exists():
        _migrate_csv_if_needed(csv_path)

    file_exists = csv_path.exists()
    with open(csv_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=_FIELDNAMES)
        if not file_exists:
            writer.writeheader()
        writer.writerow(
            {
                "checkpoint": int(iteration),
                "trained_timesteps": int(trained_timesteps),
                "eval_steps": result.steps,
                "eval_return": result.eval_return,
                "final_xy_dist": result.final_xy_dist,
                "initial_xy_dist": result.initial_xy_dist,
                "reached_target": result.reached_target,
            }
        )

    return csv_path
