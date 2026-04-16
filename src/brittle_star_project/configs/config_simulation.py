from dataclasses import dataclass
from typing import Optional

from brittle_star_project.environment.env_types import Backend


@dataclass
class SimulationSettings:
    """Settings for the simulation script."""

    model_path: Optional[str] = None
    model_type: str = "random"
    backend: Backend = Backend.MJC

    # Script behavior
    headless: bool = False
    # If None, viewer mode runs until window closed or target reached.
    max_steps: Optional[int] = None
