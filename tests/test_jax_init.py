import jax


def test_jax_initializes():
    """Verify that JAX initializes and exposes at least one device."""
    devices = jax.devices()
    assert len(devices) > 0, "JAX should expose at least one device"
