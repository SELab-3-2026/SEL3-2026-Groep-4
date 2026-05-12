import flax.struct
import jax.numpy as jnp


@flax.struct.dataclass
class EpisodeStatistics:
    episode_returns: jnp.ndarray
    episode_lengths: jnp.ndarray
    returned_episode_returns: jnp.ndarray
    returned_episode_lengths: jnp.ndarray
