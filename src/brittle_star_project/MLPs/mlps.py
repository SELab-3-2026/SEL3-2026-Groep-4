from dataclasses import dataclass, fields, field

import flax
import flax.linen as nn
import jax.tree_util
from typing import Sequence, Callable
from flax.linen.initializers import constant, orthogonal


# semi generic so we can easily make a config for it in experiments
class GenericDenseLayersWithActivation(nn.Module):
    layer_sizes: Sequence[int] = field(default_factory=lambda: [64, 64])
    activation: Callable = nn.tanh

    def setup(self):
        self.dense_layers = [nn.Dense(size) for size in self.layer_sizes]

    @nn.compact
    def __call__(self, x):
        for layer in self.dense_layers:
            x = layer(x)
            x = self.activation(x)
        return x


class OneDenseLayerMLP(nn.Module):
    feature_dim: int = field(default=1)

    def setup(self):
        self.dense_layer = nn.Dense(self.feature_dim)

    @nn.compact
    def __call__(self, x):
        return self.dense_layer(x)


class Actor(nn.Module):
    action_dim: int

    @nn.compact
    def __call__(self, x):
        mean = nn.Dense(self.action_dim, kernel_init=orthogonal(0.01), bias_init=constant(0.0))(x)
        log_std = self.param("log_std", nn.initializers.zeros, (self.action_dim,))
        return mean, log_std


@jax.tree_util.register_dataclass
@dataclass
class AgentParams:
    sensor_params: flax.core.FrozenDict
    actor_params: flax.core.FrozenDict
    critic_params: flax.core.FrozenDict
    feature_extractor_params: flax.core.FrozenDict


@jax.tree_util.register_dataclass
@dataclass
class Storage:
    obs: jax.Array
    actions: jax.Array
    logprobs: jax.Array
    dones: jax.Array
    values: jax.Array
    advantages: jax.Array
    returns: jax.Array
    rewards: jax.Array

    def replace(self, **kwargs) -> "Storage":
        fs = fields(self)
        return Storage(**{f.name: kwargs.get(f.name, getattr(self, f.name)) for f in fs})
