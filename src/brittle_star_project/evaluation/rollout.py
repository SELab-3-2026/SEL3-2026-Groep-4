from __future__ import annotations

import itertools
import time
from dataclasses import dataclass
from typing import Any

import numpy as np

from brittle_star_project import BrittleStarEnv
from brittle_star_project.evaluation.policy import ControlPolicy


@dataclass
class EpisodeResult:
    return_: float
    length: int
    reached_target: bool
    final_xy_dist: float | None
    initial_target_distance: float | None


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


def rollout_headless(
    *,
    env: BrittleStarEnv,
    policy: ControlPolicy,
    seed: int,
    max_steps: int,
    action_low: np.ndarray | None,
    action_high: np.ndarray | None,
    action_mask: np.ndarray | None = None,
) -> EpisodeResult:
    """Run an episode headlessly and return the result."""
    state = env.reset(seed=seed)

    ep_return = 0.0
    observations = _get_observations(state)
    prev_dist = _get_xy_distance_to_target(observations) if observations else None
    initial_target_distance = prev_dist
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
        cur_dist = _get_xy_distance_to_target(observations) if observations else None
        if prev_dist is not None and cur_dist is not None:
            ep_return += prev_dist - cur_dist
        prev_dist = cur_dist

        reached_target = _target_reached(state=state)
        if reached_target:
            break

    final_dist = _get_xy_distance_to_target(observations) if observations else None
    return EpisodeResult(
        return_=ep_return,
        length=steps,
        reached_target=reached_target,
        final_xy_dist=final_dist,
        initial_target_distance=initial_target_distance,
    )


def rollout_viewer(
    *,
    env: BrittleStarEnv,
    policy: ControlPolicy,
    seed: int,
    state: Any,
    control_dt: float,
    max_steps: int | None,
    action_low: np.ndarray | None,
    action_high: np.ndarray | None,
    action_mask: np.ndarray | None = None,
) -> None:
    """Run an episode using the interactive MuJoCo viewer."""
    import mujoco.viewer

    model = state.mj_model
    data = state.mj_data

    episode_return = 0.0
    observations = _get_observations(state)
    prev_dist = _get_xy_distance_to_target(observations) if observations else None
    reached_target = _target_reached(state=state)

    steps = 0
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

            with viewer.lock():
                state = env.step(state=state, action=action)

            if not viewer.is_running():
                break
            viewer.sync()

            steps += 1

            observations = _get_observations(state)
            cur_dist = _get_xy_distance_to_target(observations) if observations else None
            if prev_dist is not None and cur_dist is not None:
                episode_return += prev_dist - cur_dist
            prev_dist = cur_dist

            reached_target = _target_reached(state=state)
            if reached_target:
                break

            remaining = control_dt - (time.time() - step_start)
            if remaining > 0:
                time.sleep(remaining)

    dist = _get_xy_distance_to_target(observations) if observations else None
    dist_str = "n/a" if dist is None else f"{dist:.3f}"
    print(
        "episode done: "
        f"return={episode_return:.6f}, len={steps}, "
        f"target_reached={reached_target}, final_xy_dist={dist_str}"
    )
