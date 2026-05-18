"""Shared JAX routing utilities for decentralized multi-agent models."""

import jax


def apply_per_node(apply_fn, params, x):
    """Apply a Flax module independently to each node.

    Args:
        apply_fn: The module's ``apply`` method (e.g. ``sensor.apply``).
        params: Per-node parameters with shape ``(num_nodes, ...)``.
        x: Input tensor with shape ``(batch, num_nodes, features)``.

    Returns:
        Output tensor with shape ``(batch, num_nodes, out_features)``.
    """

    def apply_single_node(p, x_node):
        # x_node: (batch, feat) — one node's input across the batch
        return jax.vmap(lambda xi: apply_fn(p, xi))(x_node)

    return jax.vmap(apply_single_node, in_axes=(0, 1), out_axes=1)(params, x)
