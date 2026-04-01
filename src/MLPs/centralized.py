from typing import Sequence
import flax.linen as nn
import jax.numpy as jnp
from flax.linen.initializers import constant, orthogonal


class Network(nn.Module):
    hidden_size: int = 256

    @nn.compact
    def __call__(self, x):
        # x shape: (batch, obs_dim)
        x = nn.Dense(self.hidden_size, kernel_init=orthogonal(jnp.sqrt(2)))(x)
        x = nn.tanh(x)
        x = nn.Dense(self.hidden_size, kernel_init=orthogonal(jnp.sqrt(2)))(x)
        x = nn.tanh(x)
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
