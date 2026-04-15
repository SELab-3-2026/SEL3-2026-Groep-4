from dataclasses import dataclass


@dataclass
class ExperimentConfig:
    exp_name: str = "brittle_star_ppo"
    seed: int = 1
    torch_deterministic: bool = True
    cuda: bool = True
    debug_sanity: bool = False
