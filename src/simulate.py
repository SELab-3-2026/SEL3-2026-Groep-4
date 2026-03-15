from __future__ import annotations

import argparse
from pathlib import Path

from brittle_star_project import (
    ArenaConfig,
    Backend,
    BrittleStarEnv,
    BrittleStarEnvFactory,
    EnvConfig,
    MorphologyConfig,
    Task,
)
from brittle_star_project.rl import RLModel, RandomPolicyModel
from brittle_star_project.renderer import SimulationConfig, simulate_policy

MODEL_OPTIONS = ["random"]
MODEL_BY_NAME = {
    "random": RandomPolicyModel,
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Simulate a trained policy in the MuJoCo viewer.")
    p.add_argument(
        "--model",
        type=str,
        default=None,
        help="Path to a saved model artifact. If omitted, a model is created from --model-type.",
    )
    p.add_argument(
        "--model-type",
        choices=MODEL_OPTIONS,
        default="random",
        help="Which model class to instantiate when --model is omitted.",
    )
    p.add_argument("--task", choices=[t.value for t in Task], default=Task.DIRECTED_LOCOMOTION.value)
    p.add_argument("--seed", type=int, default=None)
    return p.parse_args()


def main() -> None:
    args = parse_args()

    # ======= ENVIRONMENT SETUP =======

    backend = Backend.MJC
    task = Task(args.task)

    morphology_cfg = MorphologyConfig()
    arena_cfg = ArenaConfig(attach_target=(task == Task.DIRECTED_LOCOMOTION))
    env_cfg = EnvConfig(task=task)

    factory = BrittleStarEnvFactory()
    raw_env = factory.create_environment(backend, morphology_cfg, arena_cfg, env_cfg)
    env = BrittleStarEnv(raw_env, backend=backend, config=env_cfg)

    seed_for_env = int(args.seed) if args.seed is not None else 0
    state = env.reset(seed=seed_for_env)

    # ======= MODEL SETUP =======

    nu = int(state.mj_model.nu)

    if args.model is not None:
        model_path = Path(args.model)
        policy = RLModel.load(model_path)
        if hasattr(policy, "nu"):
            policy.nu = nu
    else:
        model_cls = MODEL_BY_NAME[str(args.model_type)]
        policy = model_cls(seed=seed_for_env)
        if hasattr(policy, "nu"):
            policy.nu = nu

    default_seed = int(getattr(policy, "seed", seed_for_env))
    if args.seed is not None and hasattr(policy, "reset"):
        policy.reset(int(args.seed))

    # ======= SIMULATION =======

    rollout_cfg = SimulationConfig(
        realtime=True,
        seed=int(args.seed) if args.seed is not None else default_seed,
    )

    simulate_policy(env.raw, state=state, config=rollout_cfg, policy=policy)
    env.close()


if __name__ == "__main__":
    main()
