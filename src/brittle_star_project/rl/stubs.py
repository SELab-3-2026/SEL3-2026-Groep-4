from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from .base import RLAlgorithm


@dataclass(slots=True)
class RandomAlgorithm(RLAlgorithm):
    """A placeholder algorithm that samples random actions.

    Works if the environment exposes either:
    - a Gymnasium-style `action_space`
    - or `action_size` / `nu` integer attributes

    This is just to validate the training loop wiring.
    """

    env: Any

    def select_action(self, *, obs: Any, rng: Any | None = None) -> Any:
        if hasattr(self.env, "action_space"):
            return self.env.action_space.sample()

        for attr in ("action_size", "nu"):
            if hasattr(self.env, attr):
                n = int(getattr(self.env, attr))
                return np.random.uniform(low=-1.0, high=1.0, size=(n,)).astype(np.float32)

        # Fall back to a scalar action.
        return np.float32(0.0)

    def save(self, path: str) -> None:
        raise NotImplementedError("RandomAlgorithm has no parameters to save")

    def load(self, path: str) -> None:
        raise NotImplementedError("RandomAlgorithm has no parameters to load")


@dataclass(slots=True)
class ZeroControlPolicy:
    """For low-level MuJoCo model stepping, always applies zero controls."""

    def ctrl(self, *, nu: int, t: float) -> np.ndarray:
        return np.zeros((nu,), dtype=np.float32)
