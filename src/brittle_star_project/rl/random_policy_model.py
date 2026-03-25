from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .base import RLModel, register_rl_model


@register_rl_model("random")
@dataclass(slots=True)
class RandomPolicyModel(RLModel):
    """A minimal, serializable policy model that outputs random controls.

    This is intentionally *not* a learning algorithm yet. It exists so we can:
    - produce a stable model artifact from `train.py`
    - load that artifact in `simulate.py`
    - drive the MuJoCo viewer with the model's actions
    """

    nu: int = 0
    seed: int = 0
    ctrl_noise_scale: float = 0.5

    _rng: np.random.RandomState = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self.reset(self.seed)

    def reset(self, seed: int | None = None) -> None:
        if seed is not None:
            self.seed = int(seed)
        self._rng = np.random.RandomState(self.seed)

    def act(self, *, obs: np.ndarray | None = None, t: float = 0.0) -> np.ndarray:
        if self.nu <= 0:
            return np.zeros((0,), dtype=np.float32)
        ctrl = self.ctrl_noise_scale * self._rng.randn(self.nu)
        return ctrl.astype(np.float32)

    def to_payload(self) -> dict[str, object]:
        return {
            "seed": int(self.seed),
            "ctrl_noise_scale": float(self.ctrl_noise_scale),
        }

    @classmethod
    def from_payload(cls, payload: dict[str, object]) -> RandomPolicyModel:
        return cls(
            seed=int(payload.get("seed", 0)),
            ctrl_noise_scale=float(payload.get("ctrl_noise_scale", 0.5)),
        )
