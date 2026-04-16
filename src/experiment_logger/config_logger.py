from dataclasses import dataclass
from typing import Optional


@dataclass
class LoggingConfig:
    track: bool = False
    wandb_project_name: str = "PPO-Modularity"
    wandb_entity: Optional[str] = "SEL3-2026-Groep-4"
    capture_video: bool = False
    save_model: bool = True
    checkpoint_frequency: int = 100
    upload_model: bool = False
    hf_entity: str = ""
