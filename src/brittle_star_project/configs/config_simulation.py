from dataclasses import dataclass
from typing import Optional


@dataclass
class SimulationSettings:
    """Settings for the simulation script."""

    model_path: Optional[str] = None

    # Script behavior
    headless: bool = False
    # If None, viewer mode runs until window closed or target reached.
    max_steps: Optional[int] = None

    # Override morphology for amputation experiments.
    # When set, the environment uses this morphology instead of the trained one.
    # Points to a morphology config YAML file (e.g. configs/morphology/3_arms.yaml).
    # Observations are padded from the override morphology UP TO the training
    # morphology's shape via compute_padding_masks(override, reference=training).
    morphology_override: Optional[str] = None

    # Video recording (requires [evaluation] extra)
    record_video: bool = False
    # When None, video is saved in a per-model evaluation folder alongside the model.
    video_output_path: Optional[str] = None
