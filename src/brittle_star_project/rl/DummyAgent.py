from dataclasses import dataclass, fields
from functools import partial
from typing import Sequence

import flax
import flax.linen as nn
import jax.numpy as jnp
import jax.tree_util
import numpy as np
from flax.linen.initializers import constant, orthogonal


# inspired by: https://github.com/vwxyzjn/cleanrl/blob/master/cleanrl/ppo_atari_envpool_xla_jax_scan.py
class Network(nn.Module):
    hidden_dim: int = 195

    @nn.compact
    def __call__(self, x):
        x = nn.Dense(self.hidden_dim, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0))(x)
        x = nn.relu(x)
        x = nn.Dense(self.hidden_dim, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0))(x)
        x = nn.relu(x)
        return x


class Critic(nn.Module):
    @nn.compact
    def __call__(self, x):
        return nn.Dense(1, kernel_init=orthogonal(1), bias_init=constant(0.0))(x)


class Actor(nn.Module):
    action_dim: int

    @nn.compact
    def __call__(self, x):
        mean = nn.Dense(self.action_dim, kernel_init=orthogonal(0.01), bias_init=constant(0.0))(x)
        log_std = self.param('log_std', nn.initializers.zeros, (self.action_dim,))
        return mean, log_std


@jax.tree_util.register_dataclass
@dataclass
class AgentParams:
    network_params: flax.core.FrozenDict
    actor_params: flax.core.FrozenDict
    critic_params: flax.core.FrozenDict

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
