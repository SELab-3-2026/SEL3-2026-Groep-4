import jax
from experiment_logger import get_logger

if __name__ == "__main__":
    logger = get_logger()
    logger.info(f"JAX devices: {jax.devices()}")
