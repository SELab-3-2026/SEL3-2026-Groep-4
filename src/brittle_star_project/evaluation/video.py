from __future__ import annotations

import datetime
from pathlib import Path

import numpy as np
import yaml

from brittle_star_project import BrittleStarEnv
from brittle_star_project.evaluation.policy import ControlPolicy
from brittle_star_project.evaluation.rollout import (
    EpisodeResult,
    _get_observations,
    _get_xy_distance_to_target,
    _target_reached,
    _maybe_clip_action,
)


def create_evaluation_dir(model_path: Path) -> Path:
    """Create a unique timestamped directory for saving evaluation results."""
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    eval_dir = model_path.parent / f"{model_path.stem}_evaluations" / f"eval_{timestamp}"
    eval_dir.mkdir(parents=True, exist_ok=True)
    return eval_dir


def save_evaluation_metadata(
    eval_dir: Path,
    *,
    morphology_override_path: str | None,
    seed: int,
    max_steps: int | None,
    result: EpisodeResult,
) -> None:
    """Save metadata about the evaluation run."""
    metadata = {
        "timestamp": datetime.datetime.now().isoformat(),
        "morphology_override": morphology_override_path,
        "seed": seed,
        "max_steps": max_steps,
        "result": {
            "return": float(result.return_),
            "length": int(result.length),
            "reached_target": bool(result.reached_target),
            "final_xy_dist": float(result.final_xy_dist)
            if result.final_xy_dist is not None
            else None,
        },
    }
    with open(eval_dir / "evaluation_metadata.yaml", "w") as f:
        yaml.safe_dump(metadata, f, sort_keys=False)


def record_episode(
    *,
    env: BrittleStarEnv,
    policy: ControlPolicy,
    seed: int,
    max_steps: int,
    action_low: np.ndarray | None,
    action_high: np.ndarray | None,
    action_mask: np.ndarray | None = None,
    output_path: Path,
    fps: int = 60,
    width: int = 640,
    height: int = 480,
) -> EpisodeResult:
    """Run an episode headlessly and record a video using MuJoCo's Renderer and imageio.

    Args:
        env: The environment.
        policy: The policy agent.
        seed: Random seed.
        max_steps: Maximum number of steps.
        action_low: Minimum action values.
        action_high: Maximum action values.
        action_mask: Boolean mask for the actions.
        output_path: Where to save the .mp4 file.
        fps: Frames per second for the video.
        width: Video width.
        height: Video height.
    """
    try:
        import imageio
        import mujoco
    except ImportError as e:
        raise ImportError(
            "Video recording requires 'imageio' and 'mujoco'. "
            "Please install the evaluation dependencies: `uv pip install .[evaluation]`"
        ) from e

    state = env.reset(seed=seed)
    model = state.mj_model
    data = state.mj_data

    # Use the first camera defined in the environment config, or default to 0
    camera_id = env._config.camera_ids[0] if env._config.camera_ids else 0
    renderer = mujoco.Renderer(model, width=width, height=height)

    ep_return = 0.0
    observations = _get_observations(state)
    prev_dist = _get_xy_distance_to_target(observations) if observations else None
    reached_target = _target_reached(state=state)

    frames = []
    steps = 0

    for _ in range(int(max_steps)):
        # Capture frame
        renderer.update_scene(data, camera=camera_id)
        frames.append(renderer.render())

        # Step environment
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

    # Capture final frame
    renderer.update_scene(data, camera=camera_id)
    frames.append(renderer.render())
    renderer.close()

    # Save video
    imageio.mimsave(str(output_path), frames, fps=fps)

    final_dist = _get_xy_distance_to_target(observations) if observations else None
    return EpisodeResult(
        return_=ep_return,
        length=steps,
        reached_target=reached_target,
        final_xy_dist=final_dist,
    )
