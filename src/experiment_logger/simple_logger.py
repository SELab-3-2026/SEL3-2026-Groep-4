"""Simple terminal logger for running without external backends.

This is used for standalone package usage where WandB or TensorBoard are not desired.
It preserves the same API as UnifiedLogger but simply prints to stdout.
"""

import logging
from typing import Any, Dict, Optional


class SimpleLogger:
    """Simple logger that implements the UnifiedLogger interface via print statements."""

    def __init__(
        self,
        run_name: str = "simple_run",
        full_config: Optional[Dict[str, Any]] = None,
        logging_cfg: Optional[Any] = None,
        base_dir: str = "runs",
        save_code: bool = False,
        log_level: int = logging.INFO,
        _set_as_global: bool = False,
    ):
        self.is_interactive = True
        self.run_name = run_name
        self.full_config = full_config or {}
        print(f"[INIT] SimpleLogger initialized for run: {run_name}")

    def set_level(self, level: int):
        pass

    def log_non_interactive(self, msg: str, *args, **kwargs):
        """In SimpleLogger, we just print everything as we assume interactive use."""
        self.info(msg, *args, **kwargs)

    def progress_bar(self, iterable=None, *args, **kwargs):
        """Standard tqdm wrapper that falls back to range if tqdm is missing."""
        try:
            import tqdm

            return tqdm.tqdm(iterable, *args, **kwargs)
        except ImportError:
            return iterable

    def info(self, msg: str, *args, **kwargs):
        print(f"[INFO] {msg}")

    def warning(self, msg: str, *args, **kwargs):
        print(f"[WARNING] {msg}")

    def error(self, msg: str, *args, **kwargs):
        print(f"[ERROR] {msg}")

    def debug(self, msg: str, *args, **kwargs):
        print(f"[DEBUG] {msg}")

    def log(self, metrics: Dict[str, Any], step: Optional[int] = None, commit: bool = True):
        step_str = f"Step {step}" if step is not None else "Log"
        metric_str = ", ".join(f"{k}: {v}" for k, v in metrics.items())
        print(f"[{step_str}] {metric_str}")

    def save_checkpoint(
        self,
        params: Any,
        step: int,
        prefix: str = "checkpoint",
        metadata: Optional[Dict[str, Any]] = None,
    ):
        print(f"[SAVE] Checkpoint '{prefix}' would be saved at step {step} (SimpleLogger: No-Op)")

    def save_final_model(self, params: Any, metadata: Optional[Dict[str, Any]] = None):
        print("[SAVE] Final model would be saved (SimpleLogger: No-Op)")

    def sync_file(self, path: Any):
        """No-op for SimpleLogger."""
        pass

    def finish(self):
        print(f"[FINISH] SimpleLogger finished for run: {self.run_name}")

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.finish()
