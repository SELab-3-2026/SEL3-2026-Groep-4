"""Unified logger that writes to multiple backends simultaneously.

This logger ensures all experimental data is preserved by writing to:
1. Weights & Biases (when available)
2. Local disk (JSON files, model checkpoints, run.log)
3. stdout (for real-time monitoring)
"""

import datetime
import logging
import subprocess
import yaml
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import flax
import jax.numpy as jnp
import numpy as np

from experiment_logger.wandb_utils import finish_wandb, init_wandb

# Global storage for the active logger and the proxy singleton
_active_logger: Optional[Any] = None
_proxy_instance: Optional["LoggerProxy"] = None


def get_logger() -> "LoggerProxy":
    """Retrieve the global LoggerProxy.

    This should be used for all logging calls. It returns a proxy that
    delegates to the active logger (defaulting to a SimpleLogger until
    init_logger is called).
    """
    global _proxy_instance, _active_logger
    if _proxy_instance is None:
        if _active_logger is None:
            # Fallback to SimpleLogger to avoid premature directory creation
            from experiment_logger.simple_logger import SimpleLogger

            _active_logger = SimpleLogger(run_name="pre_init")

        _proxy_instance = LoggerProxy()

    return _proxy_instance


def init_logger(**kwargs) -> "UnifiedLogger":
    """Initialize the full UnifiedLogger and set it as the active logger.

    This should be called once the configuration is ready. It will create
    the output directories and set up all logging backends.
    """
    global _active_logger
    logger = UnifiedLogger(_set_as_global=False, **kwargs)
    _active_logger = logger
    return logger


class LoggerProxy:
    """Proxy that delegates all method calls to the active logger instance.

    This allows the logger to be swapped out (e.g., from a SimpleLogger to
    a UnifiedLogger) without any clients needing to update their references.
    """

    def _get_logger(self) -> Any:
        global _active_logger
        if _active_logger is None:
            # This shouldn't normally happen since get_logger handles it
            from experiment_logger.simple_logger import SimpleLogger

            _active_logger = SimpleLogger(run_name="pre_init_fallback")
        return _active_logger

    def __getattr__(self, name: str) -> Any:
        return getattr(self._get_logger(), name)

    def __enter__(self):
        return self._get_logger().__enter__()

    def __exit__(self, exc_type, exc_val, exc_tb):
        return self._get_logger().__exit__(exc_type, exc_val, exc_tb)


class UnifiedLogger:
    """Unified logger for scientific experiments with redundant backup."""

    def __init__(
        self,
        run_name: str,
        config: Dict[str, Any],
        project_name: str = "PPO-Modularity",
        entity: Optional[str] = None,
        base_dir: str = "runs",
        use_wandb: bool = True,
        save_code: bool = True,
        log_level: int = logging.INFO,
    ):
        """Initialize the unified logger.

        Args:
            run_name: Unique name for this run
            config: Configuration dictionary with hyperparameters
            project_name: WandB project name
            entity: WandB entity (team/user name)
            base_dir: Base directory for local storage
            use_wandb: Whether to use WandB logging
            save_code: Whether to save code to WandB
        """
        self.run_name = run_name
        self.config = config
        self.use_wandb = use_wandb
        self.wandb_available = False
        self.wandb_run = None
        self.is_interactive = sys.stdout.isatty()

        # Setup local storage
        self.run_dir = Path(base_dir) / run_name
        self.run_dir.mkdir(parents=True, exist_ok=True)

        self.checkpoints_dir = self.run_dir / "checkpoints"
        self.checkpoints_dir.mkdir(exist_ok=True)

        self.metrics_dir = self.run_dir / "metrics"
        self.metrics_dir.mkdir(exist_ok=True)

        self.config_file = self.run_dir / "config.yaml"

        # Setup standard Python logging mirror
        self.text_log_file = self.run_dir / "run.log"
        self._text_logger = logging.getLogger(f"UnifiedLogger_{self.run_name}")
        self._text_logger.setLevel(log_level)
        self._text_logger.propagate = False

        # Avoid duplicate handlers if re-instantiated
        if not self._text_logger.handlers:
            fh = logging.FileHandler(self.text_log_file)
            ch = logging.StreamHandler()

            formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
            fh.setFormatter(formatter)
            ch.setFormatter(formatter)

            self._text_logger.addHandler(fh)
            self._text_logger.addHandler(ch)

        # Save config to disk
        self._save_config()

        # Setup TensorBoard
        self.writer = None
        try:
            from torch.utils.tensorboard import SummaryWriter

            self.writer = SummaryWriter(self.run_dir)
            self.info("TensorBoard SummaryWriter initialized.")
        except ImportError:
            self.warning("tensorboard not installed. Skipping SummaryWriter.")

        # Initialize WandB if requested
        if self.use_wandb:
            self._init_wandb(project_name, entity, save_code)

        # Initialize metrics storage
        self.metrics_buffer: List[Dict[str, Any]] = []
        self.step_counter = 0

        self.info(f"Initialized UnifiedLogger for run: {run_name}")
        self.info(f"Local storage: {self.run_dir.absolute()}")
        self.info(f"WandB logging: {self.wandb_available}")

    def set_level(self, level: int):
        """Dynamically update the verbosity of the stdout/text logger."""
        self._text_logger.setLevel(level)

    def log_non_interactive(self, msg: str, *args, **kwargs):
        """Log an info message only if running in a non-interactive environment."""
        if not self.is_interactive:
            self.info(msg, *args, **kwargs)

    def progress_bar(self, iterable=None, *args, **kwargs):
        """Wrapper around tqdm that automatically disables in non-interactive environments."""
        import tqdm

        kwargs.setdefault("disable", not self.is_interactive)
        return tqdm.tqdm(iterable, *args, **kwargs)

    def info(self, msg: str, *args, **kwargs):
        """Log an info message to stdout and disk."""
        self._text_logger.info(msg, *args, **kwargs)

    def warning(self, msg: str, *args, **kwargs):
        """Log a warning message to stdout and disk."""
        self._text_logger.warning(msg, *args, **kwargs)

    def error(self, msg: str, *args, **kwargs):
        """Log an error message to stdout and disk."""
        self._text_logger.error(msg, *args, **kwargs)

    def debug(self, msg: str, *args, **kwargs):
        """Log a debug message to stdout and disk."""
        self._text_logger.debug(msg, *args, **kwargs)

    def _init_wandb(self, project_name: str, entity: Optional[str], save_code: bool):
        """Initialize Weights & Biases logging."""
        self.wandb_run = init_wandb(
            project=project_name,
            entity=entity,
            name=self.run_name,
            config=self.config,
            save_code=save_code,
            resume="allow",
        )
        self.wandb_available = self.wandb_run is not None

    def _save_config(self):
        """Save configuration to disk."""
        try:
            with open(self.config_file, "w") as f:
                yaml.dump(self.config, f, default_flow_style=False, indent=2, sort_keys=False)
            self.info(f"Config saved to {self.config_file}")
        except Exception as e:
            self.error(f"Error saving config: {e}")

    def log(self, metrics: Dict[str, Any], step: Optional[int] = None, commit: bool = True):
        """Log metrics to all backends.

        Args:
            metrics: Dictionary of metric name -> value
            step: Global step counter (auto-incremented if None)
            commit: Whether to commit to WandB immediately
        """
        if step is None:
            step = self.step_counter
            self.step_counter += 1

        # Add timestamp
        metrics_with_metadata = {
            "step": step,
            "timestamp": time.time(),
            **metrics,
        }

        # Log to stdout
        self._log_to_stdout(metrics_with_metadata)

        # Log to WandB
        if self.wandb_run is not None:
            try:
                self.wandb_run.log(metrics, step=step, commit=commit)
            except Exception as e:
                self.warning(f"WandB logging failed: {e}")

        # Log to TensorBoard
        if self.writer is not None:
            for k, v in metrics.items():
                if isinstance(v, (int, float, np.floating, np.integer)):
                    self.writer.add_scalar(k, v, step)
                elif hasattr(v, "item"):
                    self.writer.add_scalar(k, v.item(), step)
                elif isinstance(v, (np.ndarray, jnp.ndarray)) and v.size == 1:
                    self.writer.add_scalar(k, v.item(), step)

        # Buffer for disk storage
        self.metrics_buffer.append(metrics_with_metadata)

        # Periodically flush to disk
        if len(self.metrics_buffer) >= 100:
            self._flush_metrics()

    def _log_to_stdout(self, metrics: Dict[str, Any]):
        """Log metrics to stdout for real-time monitoring."""
        step = metrics.get("step", "?")
        metric_str = ", ".join(
            f"{k}={v:.6f}" if isinstance(v, (float, np.floating)) else f"{k}={v}"
            for k, v in metrics.items()
            if k not in ["step", "timestamp"]
        )
        self.info(f"[Step {step}] {metric_str}")

    def _flush_metrics(self):
        """Flush buffered metrics to disk."""
        if not self.metrics_buffer:
            return

        try:
            metrics_file = self.metrics_dir / "metrics.yaml"
            with open(metrics_file, "a") as f:
                for metric in self.metrics_buffer:
                    # Convert numpy/jax types to native Python types for YAML serialization
                    serializable_metric = {}
                    for k, v in metric.items():
                        if hasattr(v, "item"):  # numpy/jax scalar
                            serializable_metric[k] = v.item()
                        elif isinstance(v, (np.ndarray, jnp.ndarray)):
                            serializable_metric[k] = v.tolist()
                        else:
                            serializable_metric[k] = v
                    f.write("---\n")
                    yaml.dump(serializable_metric, f, default_flow_style=False)
            self.metrics_buffer.clear()
        except Exception as e:
            self.error(f"Error flushing metrics: {e}")

    def save_checkpoint(
        self,
        params: Any,
        step: int,
        prefix: str = "checkpoint",
        metadata: Optional[Dict[str, Any]] = None,
    ):
        """Save model checkpoint to disk and optionally to WandB."""
        checkpoint_name = f"{prefix}_step_{step}.flax"
        checkpoint_path = self.checkpoints_dir / checkpoint_name

        try:
            # Save to disk using Flax serialization
            with open(checkpoint_path, "wb") as f:
                f.write(flax.serialization.to_bytes(params))

            # Save metadata if provided
            if metadata:
                metadata_path = self.checkpoints_dir / f"{prefix}_step_{step}_metadata.yaml"
                with open(metadata_path, "w") as f:
                    yaml.dump(metadata, f, default_flow_style=False)

            self.info(f"Checkpoint saved: {checkpoint_path}")

            # Log to WandB as artifact
            if self.wandb_run is not None:
                try:
                    import wandb

                    artifact = wandb.Artifact(
                        name=f"{self.run_name}_{prefix}",
                        type="model",
                        metadata=metadata or {},
                    )
                    artifact.add_file(str(checkpoint_path))
                    if metadata:
                        artifact.add_file(str(metadata_path))
                    self.wandb_run.log_artifact(artifact)
                    self.info("Checkpoint uploaded to WandB")
                except Exception as e:
                    self.warning(f"Could not upload checkpoint to WandB: {e}")

        except Exception as e:
            self.error(f"Error saving checkpoint: {e}")

    def save_final_model(self, params: Any, metadata: Optional[Dict[str, Any]] = None):
        """Save the final trained model."""
        final_model_path = self.run_dir / "final_model.flax"

        try:
            with open(final_model_path, "wb") as f:
                f.write(flax.serialization.to_bytes(params))

            if metadata:
                metadata_path = self.run_dir / "final_model_metadata.yaml"
                with open(metadata_path, "w") as f:
                    yaml.dump(metadata, f, default_flow_style=False)

            self.info(f"Final model saved: {final_model_path}")

            # Log to WandB
            if self.wandb_run is not None:
                try:
                    import wandb

                    artifact = wandb.Artifact(
                        name=f"{self.run_name}_final_model",
                        type="model",
                        metadata=metadata or {},
                    )
                    artifact.add_file(str(final_model_path))
                    if metadata:
                        artifact.add_file(str(metadata_path))
                    self.wandb_run.log_artifact(artifact)
                except Exception as e:
                    self.warning(f"Could not upload final model to WandB: {e}")

        except Exception as e:
            self.error(f"Error saving final model: {e}")

    def finish(self):
        """Finalize logging and cleanup."""
        # Flush remaining metrics
        self._flush_metrics()

        if self.writer is not None:
            self.writer.close()

        self.info(f"Run complete. Results saved to: {self.run_dir.absolute()}")

        # Finish WandB run
        if self.wandb_available:
            finish_wandb()

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.finish()
