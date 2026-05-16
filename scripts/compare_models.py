"""Compare multiple trained policies across shared evaluation conditions.

For each model listed in evaluation.comparison_models, this script runs
`comparison_num_episodes` headless rollouts (seeded sequentially from
`comparison_base_seed`) and writes a results CSV to `comparison_output_csv`.

Results include two metrics per episode:
- `eval_return`    — shaped reward (same function used during training)
- `max_velocity`  — approximated as initial_xy_dist / steps taken

Usage:
    # With the default evaluation config
    python scripts/compare_models.py evaluation=poster

    # Override the output path on the fly
    python scripts/compare_models.py evaluation=poster \\
        evaluation.comparison_output_csv=metrics/quick_comparison.csv
"""

from __future__ import annotations

import csv
import logging
import time
from pathlib import Path

import hydra
from omegaconf import DictConfig, OmegaConf

from brittle_star_project.configs.main_config import BrittleStarConfig
from brittle_star_project.configs.register_configs import register_configs
from brittle_star_project.evaluation import build_eval_env
from brittle_star_project.evaluation.checkpoint import load_metadata, metadata_to_configs
from brittle_star_project.evaluation.rollout import rollout_headless

_FIELDNAMES = [
    "model_path",
    "architecture",
    "arm_0",
    "arm_1",
    "arm_2",
    "arm_3",
    "arm_4",
    "num_active_arms",
    "seed",
    "reached_target",
    "episode_length",
    "eval_return",
    "initial_target_distance",
    "final_xy_dist",
    "approx_max_velocity",
]


def _approx_max_velocity(result) -> float | None:
    """Approximate max velocity as distance covered per step.

    This is a rough upper bound: (initial_dist - final_dist) / steps.
    """
    if result.initial_target_distance is None or result.final_xy_dist is None or result.length <= 0:
        return None
    dist_covered = result.initial_target_distance - result.final_xy_dist
    return dist_covered / result.length


@hydra.main(config_path="../configs", config_name="main_config", version_base="1.3")
def main(dict_cfg: DictConfig) -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    logger = logging.getLogger(__name__)

    cfg: BrittleStarConfig = OmegaConf.to_object(
        OmegaConf.merge(OmegaConf.structured(BrittleStarConfig), dict_cfg)
    )
    eval_cfg = cfg.evaluation

    model_paths = [str(p) for p in eval_cfg.comparison_models]
    if not model_paths:
        raise ValueError(
            "evaluation.comparison_models is empty. "
            "Add at least one model path in your evaluation config."
        )

    base_seed = int(eval_cfg.comparison_base_seed)
    num_episodes = int(eval_cfg.comparison_num_episodes)
    max_steps = int(eval_cfg.eval_max_steps)

    seeds = list(range(base_seed, base_seed + num_episodes))

    output_path = Path(hydra.utils.to_absolute_path(eval_cfg.comparison_output_csv))
    output_path.parent.mkdir(parents=True, exist_ok=True)

    logger.info(
        f"Comparing {len(model_paths)} models over {num_episodes} episodes "
        f"(seeds {seeds[0]}–{seeds[-1]})."
    )
    logger.info(f"Results will be written to: {output_path}")

    with open(output_path, "w", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=_FIELDNAMES)
        writer.writeheader()

        for model_path_str in model_paths:
            model_path = Path(hydra.utils.to_absolute_path(model_path_str))
            logger.info(f"Evaluating model: {model_path.name}")

            try:
                metadata = load_metadata(model_path)
            except FileNotFoundError as e:
                logger.warning(f"Skipping model — {e}")
                continue

            training = metadata_to_configs(metadata)

            # Determine morphologies to evaluate
            # If comparison_morphologies is empty, use the model's training morphology
            morphologies = [None]
            if eval_cfg.comparison_morphologies:
                morphologies = [
                    Path(hydra.utils.to_absolute_path(m)) for m in eval_cfg.comparison_morphologies
                ]

            for morph_path in morphologies:
                morph_label = morph_path.name if morph_path else "training"
                logger.info(f"  Morphology: {morph_label}")

                bundle = build_eval_env(
                    model_path=model_path,
                    training=training,
                    metadata=metadata,
                    morphology_override_path=morph_path,
                )

                for seed in seeds:
                    t0 = time.time()
                    result = rollout_headless(
                        env=bundle.env,
                        policy=bundle.policy,
                        seed=seed,
                        max_steps=max_steps,
                        action_low=bundle.action_low,
                        action_high=bundle.action_high,
                        action_mask=bundle.action_mask,
                    )
                    elapsed = time.time() - t0

                    velocity = _approx_max_velocity(result)

                    logger.debug(
                        f"    seed={seed:3d} | "
                        f"reached={str(result.reached_target):<5} | "
                        f"return={result.return_:+8.3f} | "
                        f"steps={result.length:4d} | "
                        f"({elapsed:.1f}s)"
                    )

                    row = {
                        "model_path": model_path_str,
                        "architecture": bundle.architecture,
                        "num_active_arms": bundle.num_active_arms,
                        "seed": seed,
                        "reached_target": result.reached_target,
                        "episode_length": result.length,
                        "eval_return": result.return_,
                        "initial_target_distance": result.initial_target_distance,
                        "final_xy_dist": result.final_xy_dist,
                        "approx_max_velocity": velocity,
                    }
                    # Add per-arm segments
                    for i, segs in enumerate(bundle.segments_per_arm):
                        row[f"arm_{i}"] = segs

                    writer.writerow(row)
                    csv_file.flush()

                bundle.env.close()

    logger.info(f"Done. Results saved to {output_path}")


if __name__ == "__main__":
    register_configs()
    main()
