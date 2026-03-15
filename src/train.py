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
from brittle_star_project.rl import RandomPolicyModel

MODEL_OPTIONS = ["random"]
MODEL_BY_NAME = {
    "random": RandomPolicyModel,
}

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Create a policy model artifact.")
    p.add_argument("--out", type=str, default="artifacts/model.json")
    p.add_argument(
        "--model-type",
        choices=MODEL_OPTIONS,
        default="random",
        help="Which model to create.",
    )
    p.add_argument("--task", choices=[t.value for t in Task], default=Task.DIRECTED_LOCOMOTION.value)
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


def main() -> None:
    args = parse_args()

    # ======= ENVIRONMENT SETUP =======
    
    backend = Backend.MJC
    task = Task(args.task)

    env_cfg = EnvConfig(task=task)
    morphology_cfg = MorphologyConfig()
    arena_cfg = ArenaConfig(attach_target=(task == Task.DIRECTED_LOCOMOTION))

    factory = BrittleStarEnvFactory(backend=backend)
    raw_env = factory.create_environment(morphology_cfg, arena_cfg, env_cfg)
    env = BrittleStarEnv(raw_env, backend=backend, config=env_cfg)

    env.reset(seed=args.seed)

    # ======= MODEL SETUP & TRAINING =======

    # Infer the model class and initialize a model instance.
    model_cls = MODEL_BY_NAME[str(args.model_type)]
    model = model_cls(seed=int(args.seed))

    # Train the model in the environment for a number of epochs.
    model.train(env=env, num_epochs=args.epochs)

    # Save the trained model parameters to a file.
    out_path = model.save(Path(args.out))
    print(f"Wrote model artifact: {out_path}")

    env.close()


if __name__ == "__main__":
    main()
