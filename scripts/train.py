import subprocess
import time

import torch
import os

from brittle_star_project.dataclasses import PPOArgs
from brittle_star_project.trainers.PPOTrainer import PPOTrainer
from brittle_star_project.environment.BrittleStarJaxEnvWrapper import BrittleStarJaxEnvWrapper

from experiment_logger import UnifiedLogger
from experiment_logger.config_utils import merge_config_with_cli, print_config


def make_env(config_path: str | None, num_envs: int) -> BrittleStarJaxEnvWrapper:
    if config_path is None:
        return BrittleStarJaxEnvWrapper.default(num_envs=num_envs)
    return BrittleStarJaxEnvWrapper.from_config(config_path, num_envs=num_envs)


def parse_args() -> PPOArgs:
    import argparse

    # Use argparse to reliably extract just the config path without swallowing --help
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--hyperparameter-config-path", type=str, default=None)
    known_args, _ = parser.parse_known_args()

    args = merge_config_with_cli(PPOArgs, config_file=known_args.hyperparameter_config_path)
    return args


def get_git_hash() -> str:
    try:
        return (
            subprocess.check_output(["git", "rev-parse", "--short", "HEAD"]).decode("ascii").strip()
        )
    except (subprocess.CalledProcessError, UnicodeDecodeError):
        return "none"


if __name__ == "__main__":
    args = parse_args()

    args.batch_size = args.num_envs * args.num_steps
    args.minibatch_size = args.batch_size // args.num_minibatches
    args.num_iterations = args.total_timesteps // args.batch_size

    git_hash = get_git_hash()
    run_name = f"{args.exp_name}__seed_{args.seed}__{git_hash}__{int(time.time())}"

    if args.run_dir is None:
        run_dir = f"/data/gent/465/vsc46589/runs/{run_name}"
    else:
        run_dir = args.run_dir

    os.makedirs(run_dir, exist_ok=True)

    # Initialize Global Logger
    logger = UnifiedLogger(
        config=vars(args),
        project_name=args.wandb_project_name,  # or default PPO-Modularity if missing
        run_name=run_name,
        base_dir=os.path.dirname(run_dir),
        use_wandb=args.track,
    )

    print_config(args, title="PPO Training Configuration")

    env = make_env(args.env_config_path, args.num_envs)
    raw_env = env.raw

    torch.backends.cudnn.deterministic = args.torch_deterministic

    ppo_trainer = PPOTrainer(args, env, run_dir, run_name)
    ppo_trainer.train()
