"""Centralized WandB initialization utilities."""

import logging
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)


def init_wandb(
    project: str,
    config: Dict[str, Any],
    name: Optional[str] = None,
    entity: Optional[str] = None,
    sync_tensorboard: bool = False,
    save_code: bool = True,
    resume: str = "allow",
    **kwargs,
):
    """Initialize WandB with standardized settings.

    This function provides a centralized way to initialize WandB across different
    scripts, ensuring consistent configuration and error handling.

    Args:
        project: WandB project name
        config: Configuration dictionary to log
        name: Run name (auto-generated if None)
        entity: WandB entity (team/user name)
        sync_tensorboard: Whether to sync tensorboard logs
        save_code: Whether to save code snapshots
        resume: Resume strategy ("allow", "must", "never", "auto")
        **kwargs: Additional arguments to pass to wandb.init()

    Returns:
        wandb.Run object if successful, None otherwise
    """
    try:
        import wandb
        import os
        import sys

        # Robust HPC checking: check for API key
        has_key = os.environ.get("WANDB_API_KEY") is not None
        if not has_key:
            try:
                # Check if logged in locally via settings/netrc
                has_key = wandb.setup().settings.api_key is not None
            except Exception:
                pass

        is_interactive = sys.stdout.isatty()

        if not has_key and not is_interactive and os.environ.get("WANDB_MODE") != "offline":
            logger.warning(
                "WANDB_API_KEY not found and environment is non-interactive. "
                "Switching to offline mode."
            )
            sync_path = f"runs/{name}" if name else "runs"
            logger.warning(f"WandB is offline. Use 'wandb sync {sync_path}' to upload logs later.")
            os.environ["WANDB_MODE"] = "offline"

        run = wandb.init(
            project=project,
            entity=entity,
            name=name,
            config=config,
            sync_tensorboard=sync_tensorboard,
            save_code=save_code,
            resume=resume,
            **kwargs,
        )
        logger.info(f"WandB initialized successfully for project '{project}', run '{run.name}'")
        return run
    except ImportError:
        logger.warning("WandB not installed. Skipping WandB initialization.")
        return None
    except Exception as e:
        logger.error(f"Failed to initialize WandB: {e}")
        return None


def finish_wandb():
    """Safely finish the current WandB run."""
    try:
        import wandb

        if wandb.run is not None:
            wandb.finish()
            logger.info("WandB run finished successfully")
    except Exception as e:
        logger.warning(f"Error finishing WandB run: {e}")
