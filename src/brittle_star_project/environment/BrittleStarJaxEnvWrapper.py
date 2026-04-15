import jax
import jax.numpy as jnp

from brittle_star_project import (
    EnvConfig,
    BrittleStarEnvFactory,
    MorphologyConfig,
    ArenaConfig,
    Backend,
)
from brittle_star_project.environment import from_file


class BrittleStarJaxEnvWrapper:
    def __init__(
        self,
        morphology: MorphologyConfig,
        arena: ArenaConfig,
        env_config: EnvConfig,
        num_envs: int,
        backend: Backend = Backend.MJX,
    ):
        self._morphology = morphology
        self._arena = arena
        self._env_config = env_config
        self._backend = backend
        self._num_envs = num_envs
        self._env = BrittleStarEnvFactory.create_environment(
            self._backend, self._morphology, self._arena, self._env_config
        )

        self._vectorized_reset = jax.jit(jax.vmap(self._env.reset))
        self._vectorized_step = jax.jit(jax.vmap(self._env.step))
        self._vectorized_action_sample = jax.jit(jax.vmap(self._env.action_space.sample))

        self._action_rng = None

        from experiment_logger import get_logger

        self.logger = get_logger()
        self.logger.info(
            f"Initialized BrittleStarJaxEnvWrapper with {num_envs} envs on {backend.value}"
        )

    @property
    def backend(self):
        return self._backend

    @property
    def raw(self):
        return self._env

    @property
    def single_action_space(self):
        return self._env.action_space

    @property
    def single_observation_space(self):
        return self._env.observation_space

    def reset(self, seed: int = 0):
        self.logger.info(f"Resetting vectorized environment environments with seed {seed}")
        self._action_rng, env_rng = jax.random.split(jax.random.PRNGKey(seed), 2)
        env_rngs = jnp.array(jax.random.split(env_rng, self._num_envs))
        return self._vectorized_reset(rng=env_rngs)

    def sample_actions(self):
        assert self._action_rng is not None, "Call reset() before sample_actions()"
        self._action_rng, *sub_rngs = jnp.array(
            jax.random.split(self._action_rng, self._num_envs + 1)
        )
        return self._vectorized_action_sample(rng=jnp.array(sub_rngs))

    def step(self, state, action):
        return self._vectorized_step(state=state, action=action)

    def close(self):
        self._env.close()

    @staticmethod
    def default(num_envs: int, backend: Backend = Backend.MJX) -> "BrittleStarJaxEnvWrapper":
        morphology = MorphologyConfig()
        arena = ArenaConfig()
        env_config = EnvConfig()
        return BrittleStarJaxEnvWrapper(
            morphology, arena, env_config, num_envs=num_envs, backend=backend
        )

    @staticmethod
    def from_config(
        config_path: str, num_envs: int, backend: Backend = Backend.MJX
    ) -> "BrittleStarJaxEnvWrapper":
        morphology_cfg, arena_cfg, env_cfg = from_file(config_path)
        return BrittleStarJaxEnvWrapper(
            morphology_cfg, arena_cfg, env_cfg, num_envs=num_envs, backend=backend
        )

    def __str__(self):
        morphology_str = str(self._morphology)
        arena_str = str(self._arena)
        env_config_str = str(self._env_config)
        return (
            f"BrittleStarJaxEnvWrapper(backend={self._backend}, num_envs={self._num_envs}, "
            + f"morphology={morphology_str}, arena={arena_str}, env_config={env_config_str})"
        )
