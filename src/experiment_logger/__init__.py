"""Unified logging framework for machine learning experiments.

This package provides a unified interface for logging to multiple backends
(WandB, disk, stdout) simultaneously, ensuring no data loss.
"""

from experiment_logger.unified_logger import UnifiedLogger
from experiment_logger.wandb_utils import finish_wandb, init_wandb

__all__ = ["UnifiedLogger", "init_wandb", "finish_wandb"]
__version__ = "0.1.0"
