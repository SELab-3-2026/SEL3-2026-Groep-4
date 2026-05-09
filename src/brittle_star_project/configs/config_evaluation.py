from __future__ import annotations

from dataclasses import dataclass


@dataclass
class EvaluationConfig:
    """Evaluation settings.

    Currently used for synchronous checkpoint evaluation during training.
    """

    # When enabled, each saved checkpoint is evaluated headlessly and the results
    # are appended to a CSV in the run's metrics/ folder.
    evaluate_checkpoints: bool = False
    eval_max_steps: int = 5000
    eval_seed: int = 0

    def __post_init__(self) -> None:
        if self.evaluate_checkpoints and self.eval_max_steps <= 0:
            raise ValueError(
                "Configuration Error: 'eval_max_steps' must be > 0 when "
                "'evaluate_checkpoints' is enabled."
            )
