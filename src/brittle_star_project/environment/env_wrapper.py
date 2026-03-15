from __future__ import annotations

import inspect
from dataclasses import dataclass
from typing import Any

import numpy as np

from .env_config import EnvConfig
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

    def __init__(self, env: Any, *, backend: Backend, config: EnvConfig) -> None:
        self._env = env
        self._backend = backend
        self._config = config

    @property
    def raw(self) -> Any:
        return self._env

    @property
    def backend(self) -> Backend:
        return self._backend

    @property
    def config(self) -> EnvConfig:
        return self._config

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

    # maybe a bit over-engineered... TODO: simplify?
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

        if isinstance(out, StepResult):
            return out

        if isinstance(out, tuple) or isinstance(out, list):
            if len(out) == 1:
                return StepResult(state=out[0])
            if len(out) == 2:
                state2, reward = out
                return StepResult(state=state2, reward=float(reward))
            if len(out) == 4:
                state2, reward, done, info = out
                return StepResult(state=state2, reward=float(reward), terminated=bool(done), info=dict(info))
            if len(out) == 5:
                state2, reward, terminated, truncated, info = out
                return StepResult(
                    state=state2,
                    reward=float(reward),
                    terminated=bool(terminated),
                    truncated=bool(truncated),
                    info=dict(info),
                )

        return StepResult(state=out)
