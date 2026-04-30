#!/usr/bin/env python3
"""Empirically extract observation bounds (focused on joint velocities).

This script creates a MuJoCo environment using the project's factory and
randomly samples actions to discover observed maxima for selected
observation keys (joint_velocity, joint_position, joint_actuator_force).

Usage:
  python scripts/extract_observation_bounds.py \
    --morphology configs/morphology/3_arms.yaml --num-steps 5000 --seed 42

If `--morphology` is omitted the default `MorphologyConfig()` is used.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import yaml
import numpy as np

from brittle_star_project import BrittleStarEnvFactory, BrittleStarEnv, Backend
from brittle_star_project.environment.env_config import (
    MorphologyConfig,
    ArenaConfig,
    EnvConfig,
)


def load_morphology(path: str | None) -> MorphologyConfig:
    if path is None:
        return MorphologyConfig()
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Morphology file not found: {p}")
    with open(p, "r") as f:
        data = yaml.safe_load(f) or {}
    return MorphologyConfig(**data)


def _extract_observations(state):
    # Under different backends the returned state may be a dict or an object
    obs = getattr(state, "observations", None)
    if obs is None and isinstance(state, dict):
        obs = state.get("observations", state)
    return obs


def find_empirical_bounds(
    morph_cfg: MorphologyConfig,
    arena_cfg: ArenaConfig,
    env_cfg: EnvConfig,
    num_steps: int = 5000,
    seed: int = 42,
) -> None:
    factory = BrittleStarEnvFactory()
    raw_env = factory.create_environment(Backend.MJC, morph_cfg, arena_cfg, env_cfg)
    env = BrittleStarEnv(raw_env, backend=Backend.MJC, config=env_cfg, morphology_config=morph_cfg)

    # Initial reset
    state = env.reset(seed=seed)

    # Determine action bounds
    action_space = getattr(raw_env, "action_space", None)
    if action_space is None:
        raise RuntimeError("Environment missing `action_space`; cannot sample actions.")

    action_low = np.asarray(action_space.low, dtype=np.float32)
    action_high = np.asarray(action_space.high, dtype=np.float32)
    action_shape = action_low.shape

    # Track maximum absolute observed values
    tracked_keys = ["joint_velocity", "joint_position", "joint_actuator_force"]
    max_observed = {k: 0.0 for k in tracked_keys}

    # Include observation at reset
    obs0 = _extract_observations(state)
    if isinstance(obs0, dict):
        for k in tracked_keys:
            if k in obs0:
                max_observed[k] = max(max_observed[k], float(np.max(np.abs(np.asarray(obs0[k])))))

    rng = np.random.RandomState(seed)
    for i in range(num_steps):
        u = rng.uniform(size=action_shape)
        action = action_low + (action_high - action_low) * u

        # Provide a numpy RNG to the env step; wrapper will pass it if accepted.
        step_out = env.step(state=state, action=action, rng=env.make_rng(seed + i + 1))

        # Unpack next state from common return conventions
        if hasattr(step_out, "state"):
            next_state = step_out.state
        elif isinstance(step_out, (tuple, list)) and len(step_out) >= 1:
            next_state = step_out[0]
        else:
            next_state = step_out

        obs = _extract_observations(next_state)
        if isinstance(obs, dict):
            for k in tracked_keys:
                if k in obs:
                    val = float(np.max(np.abs(np.asarray(obs[k]))))
                    if val > max_observed[k]:
                        max_observed[k] = val

        state = next_state

    # Print recommended bounds with a 20% safety margin
    print("\n--- Recommended Observation Bounds (20% margin) ---")
    for k, v in max_observed.items():
        if v == 0.0:
            print(f"{k}: observed max 0.0 (increase sampling or inspect env)")
        else:
            safe = v * 1.2
            print(f"{k}: [-{safe:.6f}, {safe:.6f}]  (observed max: {v:.6f})")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--morphology", type=str, default=None, help="Path to morphology YAML (optional)"
    )
    parser.add_argument(
        "--num-steps", type=int, default=5000, help="Number of random steps to sample"
    )
    parser.add_argument("--seed", type=int, default=42, help="RNG seed")
    args = parser.parse_args()

    morph_cfg = load_morphology(args.morphology)
    arena_cfg = ArenaConfig()
    env_cfg = EnvConfig()

    find_empirical_bounds(morph_cfg, arena_cfg, env_cfg, num_steps=args.num_steps, seed=args.seed)


if __name__ == "__main__":
    main()
