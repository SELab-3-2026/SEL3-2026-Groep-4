import subprocess
import time

import torch
import tyro
import yaml
import os

from brittle_star_project.dataclasses import PPOArgs
from PPOTrainer import PPOTrainer
from brittle_star_project.environment.BrittleStarJaxEnvWrapper import BrittleStarJaxEnvWrapper


def make_env(config_path: str | None, num_envs: int) -> BrittleStarJaxEnvWrapper:
    if config_path is None:
        return BrittleStarJaxEnvWrapper.default(num_envs=num_envs)
    return BrittleStarJaxEnvWrapper.from_config(config_path, num_envs=num_envs)


def parse_args() -> PPOArgs:
    temp_args = tyro.cli(PPOArgs)

    if temp_args.env_config_path is not None:
        with open(temp_args.env_config_path, "r") as f:
            config = yaml.safe_load(f)
            if config:
                # parse PPOArgs with defaults from yaml.
                for key, value in config.items():
                    if hasattr(temp_args, key):
                        setattr(temp_args, key, value)

                # Reparse CLI to ensure they OVERRIDE the yaml
                args = tyro.cli(PPOArgs, default=temp_args)
    else:
        args = temp_args
    return args


def get_git_hash() -> str:
    try:
        return (
            subprocess.check_output(["git", "rev-parse", "--short", "HEAD"]).decode("ascii").strip()
        )
    except subprocess.CalledProcessError | UnicodeDecodeError:
        return "none"


if __name__ == "__main__":
    args = parse_args()

    args.batch_size = args.num_envs * args.num_steps
    args.minibatch_size = args.batch_size // args.num_minibatches
    args.num_iterations = args.total_timesteps // args.batch_size

    git_hash = get_git_hash()
    run_name = f"{args.exp_name}__seed_{args.seed}__{git_hash}__{int(time.time())}"
    if args.run_dir is None:
        run_dir = f"runs/{run_name}"
    else:
        run_dir = args.run_dir

    os.makedirs(run_dir, exist_ok=True)

    env = make_env(args.env_config_path, args.num_envs)

    torch.backends.cudnn.deterministic = args.torch_deterministic

    ppo_trainer = PPOTrainer(args, env, run_dir, run_name)
    ppo_trainer.train()
