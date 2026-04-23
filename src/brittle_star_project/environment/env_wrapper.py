from __future__ import annotations

import inspect
from dataclasses import dataclass
from typing import Any

import numpy as np

from .env_config import EnvConfig, MorphologyConfig
from .env_types import Backend


@dataclass(slots=True)
class StepResult:
    state: Any
    reward: float | None = None
    terminated: bool | None = None
    truncated: bool | None = None
    info: dict[str, Any] | None = None


class BrittleStarEnv:
    """Thin wrapper around the underlying DualMuJoCoEnvironment.

    Goal: hide backend-specific RNG setup and provide a stable place to plug in RL.
    """

    def __init__(
        self,
        env: Any,
        *,
        backend: Backend,
        config: EnvConfig,
        morphology_config: MorphologyConfig | None = None,
    ) -> None:
        self._env = env
        self._backend = backend
        self._config = config
        self._morphology_config = morphology_config

    @property
    def raw(self) -> Any:
        return self._env

    @property
    def backend(self) -> Backend:
        return self._backend

    @property
    def config(self) -> EnvConfig:
        return self._config

    @property
    def morphology_config(self) -> MorphologyConfig | None:
        return self._morphology_config

    def make_rng(self, seed: int):
        if self._backend == Backend.MJC:
            return np.random.RandomState(seed)

        import jax

        return jax.random.PRNGKey(seed)

    def reset(self, *, seed: int = 0):
        rng = self.make_rng(seed)
        state = self._env.reset(rng=rng)
        return state

    def render(self, *, state: Any):
        return self._env.render(state=state)

    def close(self) -> None:
        self._env.close()

    def step(self, *, state: Any, action: Any, rng: Any | None = None) -> StepResult:
        """Best-effort step wrapper.

        Different env libraries return different tuples; we normalize common cases.
        """

        if not hasattr(self._env, "step"):
            raise AttributeError("Underlying env has no step() method")

        step_fn = self._env.step
        sig = inspect.signature(step_fn)
        params = list(sig.parameters)

        # Common patterns:
        # - step(state, action)
        # - step(state, action, rng)
        # - step(state, action, key)
        # We pass rng only if the callable accepts a 3rd arg.
        if len(params) >= 3 and rng is not None:
            out = step_fn(state, action, rng)
        else:
            out = step_fn(state, action)

        return out
