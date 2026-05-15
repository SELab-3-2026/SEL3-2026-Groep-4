from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from brittle_star_project.evaluation.checkpoint import load_metadata, metadata_to_configs
from brittle_star_project.evaluation.eval_env_builder import build_eval_env
from brittle_star_project.evaluation.rollout import (
    _get_observations,
    _maybe_clip_action,
    _target_reached,
)
from brittle_star_project.evaluation.video import _apply_camera_overrides, _ensure_offscreen_size


def _enum_value(enum_obj, *names: str) -> int:
    for name in names:
        if hasattr(enum_obj, name):
            return int(getattr(enum_obj, name))
    raise AttributeError(f"Could not find any of {names!r} on {enum_obj!r}")


def _hex_to_rgba(hex_color: str, alpha: float) -> np.ndarray:
    color = hex_color.lstrip("#")
    if len(color) != 6:
        raise ValueError(f"Expected a 6-digit hex color, got {hex_color!r}")
    red = int(color[0:2], 16) / 255.0
    green = int(color[2:4], 16) / 255.0
    blue = int(color[4:6], 16) / 255.0
    return np.asarray([red, green, blue, float(alpha)], dtype=np.float32)


def _append_sphere(scene, mujoco, center: np.ndarray, radius: float, rgba: np.ndarray) -> None:
    geom = scene.geoms[scene.ngeom]
    mujoco.mjv_initGeom(
        geom,
        mujoco.mjtGeom.mjGEOM_SPHERE,
        np.asarray([radius, 0.0, 0.0], dtype=np.float32),
        center,
        np.eye(3, dtype=np.float32).reshape(-1),
        rgba,
    )
    scene.ngeom += 1


def _resolve_body_id(model, body_name: str, mujoco) -> int:
    body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, body_name)
    if body_id >= 0:
        return body_id

    for fallback_name in ("BrittleStarMorphology/central_disk", "central_disk"):
        body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, fallback_name)
        if body_id >= 0:
            return body_id

    for candidate_body_id in range(1, int(model.nbody)):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, candidate_body_id)
        if name:
            return candidate_body_id

    raise ValueError(f"Body '{body_name}' not found in the model")


def main() -> None:
    parser = argparse.ArgumentParser(description="Render a static path image from a rollout.")
    parser.add_argument("model", help="Path to .flax checkpoint")
    parser.add_argument("--morphology-override", default=None)
    parser.add_argument("--output-path", required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-steps", type=int, default=5000)
    parser.add_argument("--body-name", default="BrittleStarMorphology/central_disk")
    parser.add_argument("--camera-id", type=int, default=0)
    parser.add_argument("--camera-x", type=float, default=-3.0)
    parser.add_argument("--camera-y", type=float, default=0.0)
    parser.add_argument("--camera-z", type=float, default=5.0)
    parser.add_argument("--camera-fovy", type=float, default=None)
    parser.add_argument("--target-x", type=float, default=-6.0)
    parser.add_argument("--target-y", type=float, default=0.0)
    parser.add_argument("--width", type=int, default=2160)
    parser.add_argument("--height", type=int, default=960)
    parser.add_argument("--frame-stride", type=int, default=5)
    parser.add_argument("--path-color", default="#444444")
    parser.add_argument(
        "--robot-color",
        default="#444444",
        help="Hex color for brittle star robot (e.g. #ff0000)",
    )
    args = parser.parse_args()

    model_path = Path(args.model)
    metadata = load_metadata(model_path, None)
    training = metadata_to_configs(metadata)

    bundle = build_eval_env(
        model_path=model_path,
        training=training,
        metadata=metadata,
        morphology_override_path=args.morphology_override,
    )

    if (args.target_x is None) != (args.target_y is None):
        raise ValueError("target-x and target-y must be provided together")

    target_xy = None
    if args.target_x is not None:
        target_xy = (float(args.target_x), float(args.target_y))

    try:
        import imageio
        import mujoco
    except ImportError as e:
        raise ImportError(
            "Static image rendering requires 'mujoco' and 'imageio'. "
            "Please install the evaluation dependencies: `uv pip install .[evaluation]`"
        ) from e

    reset_kwargs = {}
    if target_xy is not None:
        reset_kwargs["target_position"] = (target_xy[0], target_xy[1], 0.0)

    state = bundle.env.reset(seed=args.seed, **reset_kwargs)
    model = state.mj_model
    data = state.mj_data

    _apply_camera_overrides(
        model,
        camera_fovy={args.camera_id: float(args.camera_fovy)}
        if args.camera_fovy is not None
        else None,
        camera_xyz=(
            {args.camera_id: float(args.camera_x)} if args.camera_x is not None else None,
            {args.camera_id: float(args.camera_y)} if args.camera_y is not None else None,
            {args.camera_id: float(args.camera_z)} if args.camera_z is not None else None,
        ),
    )
    _ensure_offscreen_size(model, args.width, args.height)

    body_id = _resolve_body_id(model, args.body_name, mujoco)

    # Optionally override robot color by recoloring geoms belonging to the robot's body subtree.
    if args.robot_color is not None:
        robot_rgba = _hex_to_rgba(args.robot_color, 1.0)
        # Collect body IDs in the subtree rooted at `body_id` by walking parent links.
        nbody = int(model.nbody)
        body_parent = model.body_parentid
        robot_body_ids = set([int(body_id)])
        for i in range(1, nbody):
            cur = int(i)
            # walk up until root (0) or until we hit the robot root
            while cur not in (-1, 0, int(body_id)):
                cur = int(body_parent[cur])
            if cur == int(body_id):
                robot_body_ids.add(i)

        # Recolor geoms whose body id is in the robot subtree
        for g in range(int(model.ngeom)):
            if int(model.geom_bodyid[g]) in robot_body_ids:
                model.geom_rgba[g][:] = robot_rgba

    positions = []
    observations = _get_observations(state)

    for _ in range(int(args.max_steps)):
        positions.append(np.asarray(data.xpos[body_id], dtype=np.float32))

        obs_dict = observations or {}
        action = bundle.policy.act(observations=obs_dict)
        if bundle.action_mask is not None:
            action = action[bundle.action_mask]
        action = _maybe_clip_action(action, bundle.action_low, bundle.action_high)

        state = bundle.env.step(state=state, action=action)
        data = state.mj_data
        observations = _get_observations(state)

        if _target_reached(state=state):
            break

    positions_arr = np.vstack(positions)
    if len(positions_arr) < 2:
        raise ValueError("Need at least two rollout positions to render a path")

    path_points = positions_arr.copy()
    path_points[:, 2] -= 0.02

    path_step = max(1, int(args.frame_stride)) * 4
    path_points_visible = path_points[::path_step]
    path_rgba = _hex_to_rgba(args.path_color, 0.92)

    ctx = mujoco.GLContext(args.width, args.height)
    ctx.make_current()
    try:
        catmask = _enum_value(mujoco.mjtCatBit, "mjCAT_ALL")
        camera_type = _enum_value(mujoco.mjtCamera, "mjCAMERA_FIXED")
        font_scale = _enum_value(mujoco.mjtFontScale, "mjFONTSCALE_100")

        maxgeom = int(model.ngeom + len(path_points_visible) + 8)
        scene = mujoco.MjvScene(model, maxgeom=maxgeom)
        option = mujoco.MjvOption()
        perturb = mujoco.MjvPerturb()
        camera = mujoco.MjvCamera()
        mujoco.mjv_defaultOption(option)
        mujoco.mjv_defaultPerturb(perturb)
        mujoco.mjv_defaultCamera(camera)
        camera.type = camera_type
        camera.fixedcamid = int(args.camera_id)
        if hasattr(camera, "trackbodyid"):
            camera.trackbodyid = -1

        context = mujoco.MjrContext(model, font_scale)
        viewport = mujoco.MjrRect(0, 0, args.width, args.height)

        mujoco.mjv_updateScene(model, data, option, perturb, camera, catmask, scene)

        for idx, path_point in enumerate(path_points_visible):
            path_rgba[3] = 0.10 + 0.70 * (idx / max(len(path_points_visible) - 1, 1))
            _append_sphere(scene, mujoco, path_point, 0.03, path_rgba)

        rgb = np.empty((args.height, args.width, 3), dtype=np.uint8)
        depth = np.empty((args.height, args.width), dtype=np.float32)
        mujoco.mjr_render(viewport, scene, context)
        mujoco.mjr_readPixels(rgb, depth, viewport, context)
        imageio.imwrite(args.output_path, np.flipud(rgb))

        context.free()
    finally:
        ctx.free()

    bundle.env.close()


if __name__ == "__main__":
    main()
