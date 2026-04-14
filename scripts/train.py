import os
import time
import torch
import hydra
from omegaconf import DictConfig, OmegaConf

from brittle_star_project.configs.main_config import BrittleStarConfig
from brittle_star_project.configs.register_configs import register_configs
from brittle_star_project.dataclasses import PPOArgs
from brittle_star_project.trainers.PPOTrainer import PPOTrainer
from brittle_star_project.environment.BrittleStarJaxEnvWrapper import BrittleStarJaxEnvWrapper
from experiment_logger import init_logger, get_logger


def make_env(cfg: BrittleStarConfig) -> BrittleStarJaxEnvWrapper:
    """Create the environment using the structured configuration."""
    return BrittleStarJaxEnvWrapper(
        morphology=cfg.morphology,
        arena=cfg.arena,
        env_config=cfg.environment,
        num_envs=cfg.ppo.num_envs,
    )


def create_ppo_args_compat(cfg: BrittleStarConfig, run_dir: str) -> PPOArgs:
    """Temporary adapter to bridge BrittleStarConfig to the legacy PPOArgs.

    This will be removed in Step 4.4 once PPOTrainer is refactored.
    """
    # Flatten the hierarchical config into the expected PPOArgs format
    args = PPOArgs(
        exp_name=cfg.experiment.exp_name,
        seed=cfg.experiment.seed,
        torch_deterministic=cfg.experiment.torch_deterministic,
        cuda=cfg.experiment.cuda,
        track=cfg.logging.track,
        wandb_project_name=cfg.logging.wandb_project_name,
        wandb_entity=cfg.logging.wandb_entity,
        capture_video=cfg.logging.capture_video,
        save_model=cfg.logging.save_model,
        checkpoint_frequency=cfg.logging.checkpoint_frequency,
        upload_model=cfg.logging.upload_model,
        hf_entity=cfg.logging.hf_entity,
        total_timesteps=cfg.ppo.total_timesteps,
        learning_rate=cfg.ppo.learning_rate,
        num_envs=cfg.ppo.num_envs,
        num_steps=cfg.ppo.num_steps,
        anneal_lr=cfg.ppo.anneal_lr,
        gamma=cfg.ppo.gamma,
        gae_lambda=cfg.ppo.gae_lambda,
        num_minibatches=cfg.ppo.num_minibatches,
        update_epochs=cfg.ppo.update_epochs,
        norm_adv=cfg.ppo.norm_adv,
        clip_coef=cfg.ppo.clip_coef,
        clip_vloss=cfg.ppo.clip_vloss,
        ent_coef=cfg.ppo.ent_coef,
        vf_coef=cfg.ppo.vf_coef,
        max_grad_norm=cfg.ppo.max_grad_norm,
        target_kl=cfg.ppo.target_kl,
        run_dir=run_dir,
    )

    # Compute runtime fields
    args.batch_size = args.num_envs * args.num_steps
    args.minibatch_size = args.batch_size // args.num_minibatches
    args.num_iterations = args.total_timesteps // args.batch_size

    return args


@hydra.main(config_path="../configs", config_name="main_config", version_base="1.3")
def main(dict_cfg: DictConfig):
    # 1. Convert DictConfig to structured dataclass
    cfg: BrittleStarConfig = OmegaConf.to_object(dict_cfg)

    # 2. Setup run metadata
    # Hydra changes CWD to the output directory by default.
    # We use that as our run_dir.
    run_dir = os.getcwd()
    run_name = os.path.basename(run_dir)

    # 3. Initialize Logger
    # We pass the resolved dictionary for WandB/YAML logging
    resolved_cfg_dict = OmegaConf.to_container(dict_cfg, resolve=True, throw_on_missing=True)
    init_logger(
        run_name=run_name,
        config=resolved_cfg_dict,
        project_name=cfg.logging.wandb_project_name,
        entity=cfg.logging.wandb_entity,
        base_dir=os.path.dirname(run_dir),
        use_wandb=cfg.logging.track,
    )
    logger = get_logger()
    logger.info(f"Hydra-initialized run: {run_name}")
    logger.info(f"Output directory: {run_dir}")

    # 4. Prepare compatibility object for PPOTrainer
    ppo_args = create_ppo_args_compat(cfg, run_dir)

    # 5. Setup Environment and Torch
    env = make_env(cfg)
    torch.backends.cudnn.deterministic = cfg.experiment.torch_deterministic

    # 6. Train
    ppo_trainer = PPOTrainer(ppo_args, env, run_dir, run_name)
    ppo_trainer.train()


if __name__ == "__main__":
    register_configs()
    main()
