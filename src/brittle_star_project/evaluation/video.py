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


def _ensure_offscreen_size(model, width: int, height: int) -> None:
    vis_global = getattr(getattr(model, "vis", None), "global_", None)
    if vis_global is None:
        return
    vis_global.offwidth = int(max(width, vis_global.offwidth))
    vis_global.offheight = int(max(height, vis_global.offheight))


def _apply_camera_overrides(
    model,
    *,
    camera_fovy: dict[int, float] | None = None,
    camera_xyz: tuple[dict[int, float] | None, dict[int, float] | None, dict[int, float] | None] = (None, None, None),
) -> None:
    if not camera_fovy and not (camera_xyz[0] or camera_xyz[1] or camera_xyz[2]):
        return

    ncam = int(getattr(model, "ncam", 0))
    for cam_id, fovy in (camera_fovy or {}).items():
        if cam_id < 0 or cam_id >= ncam:
            raise ValueError(f"Camera id {cam_id} is out of range")
        model.cam_fovy[cam_id] = float(fovy)

    for cam_id, x in (camera_xyz[0] or {}).items():
        if cam_id < 0 or cam_id >= ncam:
            raise ValueError(f"Camera id {cam_id} is out of range")
        model.cam_pos[cam_id][0] = float(x)

    for cam_id, y in (camera_xyz[1] or {}).items():
        if cam_id < 0 or cam_id >= ncam:
            raise ValueError(f"Camera id {cam_id} is out of range")
        model.cam_pos[cam_id][1] = float(y)

    for cam_id, z in (camera_xyz[2] or {}).items():
        if cam_id < 0 or cam_id >= ncam:
            raise ValueError(f"Camera id {cam_id} is out of range")
        model.cam_pos[cam_id][2] = float(z)

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
    camera_id: int = 1,
    fps: int = 60,
    width: int = 640,
    height: int = 480,
    target_xy: tuple[float, float] | None = None,
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
        camera_id: Camera index to use for rendering (1 is usually close-up).
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

    state = env.reset(seed=seed, target_position=target_xy)
    model = state.mj_model
    data = state.mj_data

    _ensure_offscreen_size(model, width, height)

    renderer = mujoco.Renderer(model, width=width, height=height)
    ep_return = 0.0
    observations = _get_observations(state)
    prev_dist = _get_xy_distance_to_target(observations) if observations else None
    initial_dist = prev_dist
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
        initial_target_distance=initial_dist,
    )


def record_episode_multi_camera(
    *,
    env: BrittleStarEnv,
    policy: ControlPolicy,
    seed: int,
    max_steps: int,
    action_low: np.ndarray | None,
    action_high: np.ndarray | None,
    output_paths: dict[int, Path],
    action_mask: np.ndarray | None = None,
    camera_ids: list[int] | None = None,
    camera_fovy: dict[int, float] | None = None,
    camera_xyz: tuple[dict[int, float] | None, dict[int, float] | None, dict[int, float] | None] = (None, None, None),
    target_xy: tuple[float, float] | None = None,
    fps: int = 60,
    width: int = 640,
    height: int = 480,
) -> EpisodeResult:
    """Run one episode and render multiple camera views to separate files."""
    try:
        import imageio
        import mujoco
    except ImportError as e:
        raise ImportError(
            "Video recording requires 'imageio' and 'mujoco'. "
            "Please install the evaluation dependencies: `uv pip install .[evaluation]`"
        ) from e

    if camera_ids is None:
        camera_ids = list(output_paths.keys())

    for cam_id in camera_ids:
        if cam_id not in output_paths:
            raise ValueError(f"Missing output path for camera {cam_id}")

    output_paths = {cam_id: output_paths[cam_id] for cam_id in camera_ids}

    for path in output_paths.values():
        path.parent.mkdir(parents=True, exist_ok=True)

    state = env.reset(seed=seed, target_position=(target_xy[0], target_xy[1], 0.0))
    model = state.mj_model
    data = state.mj_data

    _apply_camera_overrides(model, camera_fovy=camera_fovy, camera_xyz=camera_xyz)

    _ensure_offscreen_size(model, width, height)

    renderer = mujoco.Renderer(model, width=width, height=height)
    frames = {cam_id: [] for cam_id in camera_ids}

    ep_return = 0.0
    observations = _get_observations(state)
    prev_dist = _get_xy_distance_to_target(observations) if observations else None
    initial_dist = prev_dist
    reached_target = _target_reached(state=state)

    steps = 0
    for _ in range(int(max_steps)):
        for cam_id in camera_ids:
            renderer.update_scene(data, camera=cam_id)
            frames[cam_id].append(renderer.render())

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

    for cam_id in camera_ids:
        renderer.update_scene(data, camera=cam_id)
        frames[cam_id].append(renderer.render())

    renderer.close()

    for cam_id, path in output_paths.items():
        imageio.mimsave(str(path), frames[cam_id], fps=fps)

    final_dist = _get_xy_distance_to_target(observations) if observations else None
    return EpisodeResult(
        return_=ep_return,
        length=steps,
        reached_target=reached_target,
        final_xy_dist=final_dist,
        initial_target_distance=initial_dist,
    )
