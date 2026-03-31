import logging

import jax

logger = logging.getLogger(__name__)

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    logger.info(f"JAX devices: {jax.devices()}")
