import os
import torch
import hydra
from omegaconf import DictConfig, OmegaConf

from brittle_star_project.configs.main_config import BrittleStarConfig
from brittle_star_project.configs.register_configs import register_configs
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


@hydra.main(config_path="../configs", config_name="main_config", version_base="1.3")
def main(dict_cfg: DictConfig):
    # 1. Convert DictConfig to structured dataclass
    cfg: BrittleStarConfig = OmegaConf.to_object(dict_cfg)

    # 2. Setup run metadata
    # Hydra changes CWD to the output directory by default.
    run_dir = os.getcwd()
    run_name = os.path.basename(run_dir)

    # 3. Initialize Logger
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

    # 4. Setup Environment and Torch
    env = make_env(cfg)
    torch.backends.cudnn.deterministic = cfg.experiment.torch_deterministic

    # 5. Train - pass structured config directly
    ppo_trainer = PPOTrainer(cfg, env, run_dir, run_name)
    ppo_trainer.train()


if __name__ == "__main__":
    register_configs()
    main()
