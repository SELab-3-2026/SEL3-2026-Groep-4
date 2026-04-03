import time

import torch
import tyro

from brittle_star_project.dataclasses import PPOArgs
from PPOTrainer import PPOTrainer
from brittle_star_project.environment.BrittleStarJaxEnvWrapper import BrittleStarJaxEnvWrapper


def make_env(config_path: str | None, num_envs: int) -> BrittleStarJaxEnvWrapper:
    if config_path is None:
        return BrittleStarJaxEnvWrapper.default(num_envs=num_envs)
    return BrittleStarJaxEnvWrapper.from_config(config_path, num_envs=num_envs)


if __name__ == "__main__":
    args = tyro.cli(PPOArgs)

    args.batch_size = args.num_envs * args.num_steps
    args.minibatch_size = args.batch_size // args.num_minibatches
    # args.num_iterations = args.total_timesteps // args.batch_size
    args.num_iterations = 5
    run_name = f"{args.exp_name}__seed_{args.seed}__{int(time.time())}"
    env = make_env(args.config_path, args.num_envs)

    torch.backends.cudnn.deterministic = args.torch_deterministic

    ppo_trainer = PPOTrainer(args, env, run_name)
    ppo_trainer.train()
