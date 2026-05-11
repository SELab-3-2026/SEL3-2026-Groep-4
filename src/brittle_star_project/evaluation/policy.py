from __future__ import annotations

from pathlib import Path
from typing import Any, Protocol

import jax
import jax.numpy as jnp
import numpy as np

from brittle_star_project.evaluation.checkpoint import load_params


class ControlPolicy(Protocol):
    """Protocol for any policy that can produce actions from observations."""

    def act(self, *, observations: dict[str, Any]) -> np.ndarray: ...


class PolicyAgent:
    """Wraps a trained Flax actor for deterministic inference."""

    def __init__(
        self,
        *,
        sensor_params: Any,
        actor_params: Any,
        message_passer_params: Any | None = None,
        message_passing_steps: int | None = None,
        adj_matrix: Any | None = None,
        action_dim: int,
        obs_processor: Any,
    ) -> None:
        from brittle_star_project.MLPs.mlps import (
            Actor,
            GenericDenseLayersWithActivation,
            MessagePasser,
        )

        # Infer layer sizes from params
        try:
            dense_params = (
                sensor_params.get("params", {})
                if isinstance(sensor_params, dict)
                else sensor_params["params"]
            )
        except Exception:
            dense_params = sensor_params

        layer_sizes = []
        idx = 0
        while True:
            key = f"Dense_{idx}"
            if key not in dense_params:
                break

            layer_sizes.append(int(np.asarray(dense_params[key]["kernel"]).shape[-1]))
            idx += 1

        if not layer_sizes:
            raise ValueError("Could not infer Dense_* layers from sensor params")

        self._sensor = GenericDenseLayersWithActivation(layer_sizes=layer_sizes)
        self._actor = Actor(action_dim=action_dim)

        self._message_passer = None
        if message_passer_params is not None and not (
            isinstance(message_passer_params, dict) and len(message_passer_params) == 0
        ):
            if message_passing_steps is None or adj_matrix is None:
                raise ValueError(
                    "Checkpoint contains message_passer_params but PolicyAgent was not given "
                    "message_passing_steps and adj_matrix. Pass these when constructing the agent "
                    "so decentralized evaluation matches training."
                )

            hidden_dim = int(layer_sizes[-1])
            self._message_passer = MessagePasser(
                hidden_dim=hidden_dim,
                num_propagation_steps=int(message_passing_steps),
                adj_matrix=jnp.asarray(adj_matrix),
            )
            self._message_passer.apply = jax.jit(self._message_passer.apply)
        self._sensor.apply = jax.jit(self._sensor.apply)
        self._actor.apply = jax.jit(self._actor.apply)
        self._params = {
            "sensor_params": sensor_params,
            "actor_params": actor_params,
            "message_passer_params": message_passer_params,
        }
        self._obs_processor = obs_processor

    @classmethod
    def from_params(
        cls,
        *,
        sensor_params: Any,
        actor_params: Any,
        message_passer_params: Any | None = None,
        message_passing_steps: int | None = None,
        adj_matrix: Any | None = None,
        action_dim: int,
        obs_processor: Any,
    ) -> "PolicyAgent":
        """Construct a PolicyAgent directly from in-memory parameters."""
        return cls(
            sensor_params=sensor_params,
            actor_params=actor_params,
            message_passer_params=message_passer_params,
            message_passing_steps=message_passing_steps,
            adj_matrix=adj_matrix,
            action_dim=action_dim,
            obs_processor=obs_processor,
        )

    def set_params(
        self,
        *,
        sensor_params: Any,
        actor_params: Any,
        message_passer_params: Any | None = None,
    ) -> None:
        """Update parameters for evaluation without rebuilding the model."""
        self._params["sensor_params"] = sensor_params
        self._params["actor_params"] = actor_params
        self._params["message_passer_params"] = message_passer_params

    @classmethod
    def from_checkpoint(
        cls,
        model_path: Path,
        *,
        action_dim: int,
        obs_processor: Any,
        message_passing_steps: int | None = None,
        adj_matrix: Any | None = None,
    ) -> "PolicyAgent":
        """Load params from .flax and construct the agent."""
        params = load_params(model_path)

        return cls(
            sensor_params=params["sensor_params"],
            actor_params=params["actor_params"],
            message_passer_params=params.get("message_passer_params"),
            message_passing_steps=message_passing_steps,
            adj_matrix=adj_matrix,
            action_dim=action_dim,
            obs_processor=obs_processor,
        )

    def _apply_per_node(self, net, params, x):
        # params: (nodes, ...)
        # x: (batch, nodes, feat)

        def apply_single_node(p, x_node):
            # x_node: (batch, feat)
            return jax.vmap(lambda xi: net.apply(p, xi))(x_node)

        return jax.vmap(apply_single_node, in_axes=(0, 1), out_axes=1)(params, x)

    def act(self, *, observations: dict[str, Any]) -> np.ndarray:
        """Return deterministic action (actor mean, no exploration noise)."""
        batched_obs = jax.tree.map(lambda x: jnp.asarray(x)[None, ...], observations)
        obs = self._obs_processor(batched_obs)

        hidden = self._apply_per_node(self._sensor, self._params["sensor_params"], obs)

        if self._message_passer is not None:
            mp_params = self._params.get("message_passer_params")
            if mp_params is None or (isinstance(mp_params, dict) and len(mp_params) == 0):
                raise ValueError(
                    "PolicyAgent has a message passer but message_passer_params are missing/empty."
                )
            hidden = jax.vmap(lambda x: self._message_passer.apply(mp_params, x))(hidden)

        mean, _log_std = self._apply_per_node(self._actor, self._params["actor_params"], hidden)

        return np.asarray(mean, dtype=np.float32).ravel()
