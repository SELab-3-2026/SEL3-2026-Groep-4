import jax
from experiment_logger import get_logger


def logged_jit(fn, **jit_kwargs):
    logger = get_logger()
    name = getattr(fn, "__name__", getattr(fn, "__qualname__", repr(fn)))

    def decorator(func):
        def traced_func(*args, **kwargs):
            logger.debug(f"[JIT] Compiling {name}...")
            return func(*args, **kwargs)

        jitted = jax.jit(traced_func, **jit_kwargs)
        return jitted

    return decorator(fn)
