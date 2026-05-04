from dataclasses import dataclass, fields, field

import flax.linen as nn
import jax.numpy as jnp
import jax.tree_util
from typing import Sequence, Callable
from flax.linen.initializers import constant, orthogonal
from flax.core import FrozenDict


# semi generic so we can easily make a config for it in experiments
class GenericDenseLayersWithActivation(nn.Module):
    layer_sizes: Sequence[int] = field(default_factory=lambda: [64, 64])
    activation: Callable = nn.tanh

    @nn.compact
    def __call__(self, x):
        for size in self.layer_sizes:
            x = nn.Dense(size, kernel_init=orthogonal(jnp.sqrt(2)))(x)
            x = self.activation(x)
        return x


class OneDenseLayerMLP(nn.Module):
    @nn.compact
    def __call__(self, x):
        return nn.Dense(1, kernel_init=orthogonal(1), bias_init=constant(0.0))(x)


class Actor(nn.Module):
    action_dim: int

    @nn.compact
    def __call__(self, x):
        mean = nn.Dense(self.action_dim, kernel_init=orthogonal(0.01), bias_init=constant(0.0))(x)
        log_std = self.param("log_std", nn.initializers.zeros, (self.action_dim,))
        return mean, log_std


class MessagePasser(nn.Module):
    hidden_dim: int
    num_propagation_steps: int

    @nn.compact
    def __call__(self, x: jnp.ndarray, adj_matrix: jnp.ndarray):
        for _ in range(self.num_propagation_steps):
            # (n_nodes, feat)
            messages = nn.Dense(self.hidden_dim)(x)
            messages = nn.tanh(messages)

            agg = adj_matrix  # if mean is wanted: adj_matrix / (adj.sum(axis=-1, keepdims=True) + 1e-8)
            aggregated = agg @ messages

            x_concat = jnp.concatenate([x, aggregated], axis=-1)

            gate = nn.sigmoid(nn.Dense(self.hidden_dim)(x_concat))
            candidate = nn.tanh(nn.Dense(self.hidden_dim)(x_concat))
            x = gate * x + (1 - gate) * candidate

        return x


@jax.tree_util.register_dataclass
@dataclass
class AgentParams:
    sensor_params: FrozenDict | dict
    actor_params: FrozenDict | dict
    critic_params: FrozenDict | dict
    feature_extractor_params: FrozenDict | dict
    message_passer_params: FrozenDict | dict


@jax.tree_util.register_dataclass
@dataclass
class Storage:
    obs: jnp.ndarray
    actions: jnp.ndarray
    logprobs: jnp.ndarray
    dones: jnp.ndarray
    values: jnp.ndarray
    advantages: jnp.ndarray
    returns: jnp.ndarray
    rewards: jnp.ndarray

    raw_actions: jnp.ndarray | None = None  # before clipping
    means: jnp.ndarray | None = None  # policy mean
    stds: jnp.ndarray | None = None  # policy std

    def replace(self, **kwargs) -> "Storage":
        fs = fields(self)
        return Storage(**{f.name: kwargs.get(f.name, getattr(self, f.name)) for f in fs})
