from __future__ import annotations

import argparse
from pathlib import Path

from brittle_star_project.evaluation.checkpoint import load_metadata, metadata_to_configs
from brittle_star_project.evaluation.eval_env_builder import build_eval_env
from brittle_star_project.evaluation.video import record_episode_multi_camera

_ARCH_DIR_MAP = {
    "CENTRALIZED": "centralized",
    "FULLY_CONNECTED": "fully-connected",
    "RING": "ring",
    "SEGMENT": "segment",
}


def _arch_dir(name: str) -> str:
    return _ARCH_DIR_MAP.get(name, name.lower())


def _resolve_overrides(overrides: list[str], count: int) -> list[str | None]:
    if not overrides:
        return [None] * count
    if len(overrides) == 1 and count > 1:
        return overrides * count
    if len(overrides) != count:
        raise ValueError("morphology overrides must match the number of models")
    return overrides


def main() -> None:
    parser = argparse.ArgumentParser(description="Render top-down and follow videos for poster.")
    parser.add_argument("models", nargs="+", help="Paths to .flax checkpoints")
    parser.add_argument(
        "--morphology-override",
        action="append",
        default=[],
        help="Override morphology YAML path (repeat to match models)",
    )
    parser.add_argument("--output-root", default="vids/poster")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-steps", type=int, required=True)
    parser.add_argument("--topdown-camera", type=int, default=0)
    parser.add_argument("--follow-camera", type=int, default=1)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--fps", type=int, default=60)
    args = parser.parse_args()

    overrides = _resolve_overrides(args.morphology_override, len(args.models))
    output_root = Path(args.output_root)

    for model_path_str, override in zip(args.models, overrides):
        model_path = Path(model_path_str)
        metadata = load_metadata(model_path, None)
        training = metadata_to_configs(metadata)

        bundle = build_eval_env(
            model_path=model_path,
            training=training,
            metadata=metadata,
            morphology_override_path=override,
        )

        arch_dir = _arch_dir(bundle.architecture)
        arms_dir = f"{bundle.num_active_arms}arms"
        out_dir = output_root / arms_dir / arch_dir
        out_dir.mkdir(parents=True, exist_ok=True)

        output_paths = {
            args.topdown_camera: out_dir / "topdown.mp4",
            args.follow_camera: out_dir / "follow.mp4",
        }

        result = record_episode_multi_camera(
            env=bundle.env,
            policy=bundle.policy,
            seed=args.seed,
            max_steps=args.max_steps,
            action_low=bundle.action_low,
            action_high=bundle.action_high,
            action_mask=bundle.action_mask,
            output_paths=output_paths,
            camera_ids=[args.topdown_camera, args.follow_camera],
            width=args.width,
            height=args.height,
            fps=args.fps,
        )

        final_dist = (
            "n/a" if result.final_xy_dist is None else f"{result.final_xy_dist:.3f}"
        )
        print(
            f"{arms_dir}/{arch_dir}: return={result.return_:.6f}, len={result.length}, "
            f"target_reached={result.reached_target}, final_xy_dist={final_dist}"
        )

        bundle.env.close()


if __name__ == "__main__":
    main()
