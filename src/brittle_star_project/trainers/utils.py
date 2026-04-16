from flax.training.train_state import TrainState
from brittle_star_project.configs.main_config import BrittleStarConfig


def serialize_training_state(cfg: BrittleStarConfig, agent_state: TrainState):
    from dataclasses import asdict as _asdict

    config_dict = {
        "experiment": _asdict(cfg.experiment),
        "ppo": _asdict(cfg.ppo),
    }
    params = [
        config_dict,
        [
            agent_state.params["sensor_params"],
            agent_state.params["actor_params"],
            agent_state.params["critic_params"],
            agent_state.params["feature_extractor_params"],
        ],
    ]
    return config_dict, params
