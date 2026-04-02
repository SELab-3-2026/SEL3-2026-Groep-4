from typing import Sequence, Callable
import flax.linen as nn
import jax.numpy as jnp
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
