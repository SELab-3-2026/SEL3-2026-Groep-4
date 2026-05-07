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
import numpy as np
from omegaconf import DictConfig, OmegaConf

from brittle_star_project import Backend, BrittleStarEnv, BrittleStarEnvFactory
from brittle_star_project.configs.main_config import BrittleStarConfig
from brittle_star_project.configs.register_configs import register_configs
from brittle_star_project.environment.obs_processing import create_obs_processor
from brittle_star_project.environment.padded_obs_wrapper import compute_padding_masks
from brittle_star_project.evaluation.checkpoint import load_metadata, metadata_to_configs
from brittle_star_project.evaluation.policy import PolicyAgent
from brittle_star_project.evaluation.rollout import rollout_headless

_FIELDNAMES = [
    "model_path",
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

            # --- Load sidecar metadata + reconstruct configs ---
            try:
                metadata = load_metadata(model_path)
            except FileNotFoundError as e:
                logger.warning(f"Skipping model — {e}")
                continue

            training = metadata_to_configs(metadata)

            # --- Build padding masks and obs_processor ---
            padding_masks = compute_padding_masks(
                segments_per_arm=training.morphology.segments_per_arm,
                reference_segments_per_arm=training.morphology.segments_per_arm,
            )
            obs_processor = create_obs_processor(
                bounds_dict=training.obs_bounds.to_bounds_dict(),
                padding_masks=padding_masks,
            )

            # --- Build the CPU environment from training config ---
            factory = BrittleStarEnvFactory()
            raw_env = factory.create_environment(
                Backend.MJC,
                training.morphology,
                training.arena,
                training.environment,
            )
            env = BrittleStarEnv(
                raw_env,
                backend=Backend.MJC,
                config=training.environment,
                morphology_config=training.morphology,
            )

            trained_action_dim = sum(training.morphology.segments_per_arm) * 2
            action_mask = np.asarray(padding_masks["mask_2x"])
            action_space = getattr(raw_env, "action_space", None)
            action_low = (
                None
                if action_space is None
                else np.asarray(action_space.low, dtype=np.float32).ravel()
            )
            action_high = (
                None
                if action_space is None
                else np.asarray(action_space.high, dtype=np.float32).ravel()
            )

            policy = PolicyAgent.from_checkpoint(
                model_path,
                action_dim=trained_action_dim,
                obs_processor=obs_processor,
            )

            # --- Run episodes ---
            for seed in seeds:
                t0 = time.time()
                result = rollout_headless(
                    env=env,
                    policy=policy,
                    seed=seed,
                    max_steps=max_steps,
                    action_low=action_low,
                    action_high=action_high,
                    action_mask=action_mask,
                )
                elapsed = time.time() - t0

                velocity = _approx_max_velocity(result)

                logger.debug(
                    f"seed={seed:3d} | "
                    f"reached={str(result.reached_target):<5} | "
                    f"return={result.return_:+8.3f} | "
                    f"steps={result.length:4d} | "
                    f"final_dist="
                    f"{'n/a' if result.final_xy_dist is None else f'{result.final_xy_dist:.3f}'} | "
                    f"({elapsed:.1f}s)"
                )

                writer.writerow(
                    {
                        "model_path": model_path_str,
                        "seed": seed,
                        "reached_target": result.reached_target,
                        "episode_length": result.length,
                        "eval_return": result.return_,
                        "initial_target_distance": result.initial_target_distance,
                        "final_xy_dist": result.final_xy_dist,
                        "approx_max_velocity": velocity,
                    }
                )
                csv_file.flush()

            env.close()

    logger.info(f"Done. Results saved to {output_path}")


if __name__ == "__main__":
    register_configs()
    main()
