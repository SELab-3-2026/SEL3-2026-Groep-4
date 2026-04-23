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

    # Optional: point to a Hydra config.yaml from a training run (e.g. runs/.../.hydra/config.yaml).
    # When set, the simulation script can override
    # morphology/arena/environment/architecture to match.
    trained_config_path: Optional[str] = None
