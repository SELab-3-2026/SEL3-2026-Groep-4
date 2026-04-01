"""Unified logging framework for machine learning experiments.

This package provides a unified interface for logging to multiple backends
(WandB, disk, stdout) simultaneously, ensuring no data loss.
"""

from experiment_logger.config_utils import load_yaml_config, merge_config_with_cli
from experiment_logger.unified_logger import UnifiedLogger, get_logger
from experiment_logger.wandb_utils import finish_wandb, init_wandb

__all__ = [
    "UnifiedLogger",
    "get_logger",
    "init_wandb",
    "finish_wandb",
    "load_yaml_config",
    "merge_config_with_cli",
]
__version__ = "0.1.0"
