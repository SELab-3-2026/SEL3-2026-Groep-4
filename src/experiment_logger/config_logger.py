from dataclasses import dataclass
from typing import Optional


@dataclass
class LoggingConfig:
    track: bool = False
    wandb_project_name: str = "default-project"
    wandb_entity: Optional[str] = "SEL3-2026-Groep-4"
    capture_video: bool = False

    # Local Saving
    save_model: bool = True  # Final model
    save_checkpoints: bool = True  # Intermediate checkpoints
    checkpoint_frequency: int = 100

    # Remote Uploading (WandB Artifacts)
    upload_final_model: bool = False
    upload_checkpoints: bool = False

    hf_entity: str = ""

    def __post_init__(self):
        if self.upload_final_model and not (self.track and self.save_model):
            raise ValueError(
                "Configuration Error: 'upload_final_model' is True, but it requires "
                "both 'track' and 'save_model' to also be True."
            )
        if self.upload_checkpoints and not (self.track and self.save_checkpoints):
            raise ValueError(
                "Configuration Error: 'upload_checkpoints' is True, but it requires "
                "both 'track' and 'save_checkpoints' to also be True."
            )
