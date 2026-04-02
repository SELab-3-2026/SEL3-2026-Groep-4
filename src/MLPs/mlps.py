from dataclasses import dataclass, fields

import flax
import flax.linen as nn
import jax.numpy as jnp
import jax.tree_util
from typing import Sequence, Callable
from flax.linen.initializers import constant, orthogonal


# example usage: network = SemiGenericNetwork(layer_sizes=[256, 256], activation=nn.relu)
# semi generic so we can easily make a config for it in experiments
class SemiGenericNetwork(nn.Module):
    layer_sizes: Sequence[int] = [64, 64]  # default 2 layers of 64 neurons
    activation: Callable = nn.tanh  # default tanh

    @nn.compact
    def __call__(self, x):
        for size in self.layer_sizes:
            x = nn.Dense(size, kernel_init=orthogonal(jnp.sqrt(2)))(x)
            x = self.activation(x)
        return x


class Critic(nn.Module):
    @nn.compact
    def __call__(self, x):
        return nn.Dense(1, kernel_init=orthogonal(1), bias_init=constant(0.0))(x)


class Actor(nn.Module):
    action_dim: Sequence[int]

    @nn.compact
    def __call__(self, x):
        return nn.Dense(self.action_dim, kernel_init=orthogonal(0.01), bias_init=constant(0.0))(x)


@jax.tree_util.register_dataclass
@dataclass
class AgentParams:
    network_params: flax.core.FrozenDict
    actor_params: flax.core.FrozenDict
    critic_params: flax.core.FrozenDict
    critic_network_params: flax.core.FrozenDict


@jax.tree_util.register_dataclass
@dataclass
class Storage:
    obs: jnp.array
    actions: jnp.array
    logprobs: jnp.array
    dones: jnp.array
    values: jnp.array
    advantages: jnp.array
    returns: jnp.array
    rewards: jnp.array

    def replace(self, **kwargs) -> "Storage":
        fs = fields(self)
        return Storage(**{f.name: kwargs.get(f.name, getattr(self, f.name)) for f in fs})
