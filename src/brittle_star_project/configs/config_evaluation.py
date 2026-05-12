from __future__ import annotations

from dataclasses import dataclass, field


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

    # Cross-model comparison settings.
    # comparison_base_seed is the starting seed for generating episode seeds.
    comparison_base_seed: int = 0
    # comparison_num_episodes controls how many target positions to evaluate for each model.
    comparison_num_episodes: int = 5
    # comparison_models lists the paths (relative to workspace root) to the .cleanrl_model files.
    comparison_models: list[str] = field(default_factory=list)
    # Path where the comparison results CSV will be saved (relative to workspace root).
    comparison_output_csv: str = "metrics/model_comparison.csv"
    # Morphology override YAML paths for cross-morphology comparison.
    # Each path points to a file in configs/morphology/ (e.g., "configs/morphology/3_arms.yaml").
    # When empty, each model is evaluated only on its training morphology.
    comparison_morphologies: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        if self.evaluate_checkpoints and self.eval_max_steps <= 0:
            raise ValueError(
                "Configuration Error: 'eval_max_steps' must be > 0 when "
                "'evaluate_checkpoints' is enabled."
            )
