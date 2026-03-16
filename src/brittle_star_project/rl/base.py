from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any


@dataclass(frozen=True, slots=True)
class Transition:
    """A minimal transition container for RL.

    This is intentionally generic because the underlying env state type may be a
    JAX pytree, a numpy struct, or something library-specific.
    """

    obs: Any
    action: Any
    reward: float
    next_obs: Any
    terminated: bool
    truncated: bool
    info: dict[str, Any] | None = None


class RLAlgorithm(ABC):
    """Insertable RL algorithm interface."""

    @abstractmethod
    def select_action(self, *, obs: Any, rng: Any | None = None) -> Any:
        raise NotImplementedError

    def observe(self, transition: Transition) -> None:
        """Optional hook to store transitions."""

    def update(self, *, rng: Any | None = None) -> dict[str, float]:
        """Optional hook to run one training update."""

        return {}

    def save(self, path: str) -> None:
        raise NotImplementedError("Save not implemented")

    def load(self, path: str) -> None:
        raise NotImplementedError("Load not implemented")


_RL_MODEL_REGISTRY: dict[str, type["RLModel"]] = {}


def registered_model_types() -> list[str]:
    return sorted(_RL_MODEL_REGISTRY)


def create_model(type_name: str, *, payload: dict[str, Any]) -> "RLModel":
    model_cls = _RL_MODEL_REGISTRY.get(type_name)
    if model_cls is None:
        known = ", ".join(sorted(_RL_MODEL_REGISTRY)) or "<none>"
        raise ValueError(f"Unknown RLModel type '{type_name}'. Known: {known}")
    return model_cls.from_payload(payload)


def get_rl_model_registry() -> dict[str, type["RLModel"]]:
    """Return a copy of the current RLModel registry.

    The registry is populated by importing concrete model modules that use the
    `@register_rl_model(...)` decorator.
    """

    return dict(_RL_MODEL_REGISTRY)


def register_rl_model(*type_names: str):
    """Decorator to register an `RLModel` for generic loading.

    Concrete model modules should apply this decorator, so `base.py` never needs
    to import concrete models (avoids circular imports).
    """

    if not type_names:
        raise TypeError("register_rl_model() requires at least one type name")

    primary = type_names[0]

    def _decorator(cls: type[RLModel]):
        for name in type_names:
            _RL_MODEL_REGISTRY[name] = cls
        cls.type_name = primary
        return cls

    return _decorator


class RLModel(ABC):
    """Serializable policy/model interface.

    This is the artifact that `train.py` writes and `simulate.py` loads.
    """

    # Overwritten by the `@register_rl_model(...)` decorator.
    type_name: str = "RLModel"

    def reset(self, seed: int | None = None) -> None:
        """Optional hook for RNG/stateful models."""

    @abstractmethod
    def act(self, *, obs: Any | None = None, t: float = 0.0) -> Any:
        raise NotImplementedError

    def train(self, *, env: Any, num_epochs: int = 1) -> None:
        """Optional training hook.

        Many models won't learn; for those this can be a no-op.
        """

        _ = (env, num_epochs)

    def to_payload(self) -> dict[str, Any]:
        """Return JSON-serializable model parameters."""

        return {}

    @classmethod
    def from_payload(cls, payload: dict[str, Any]) -> "RLModel":
        """Reconstruct a model from `to_payload()` output."""

        return cls(**payload)  # type: ignore[arg-type]

    def save(self, path: str | Path) -> Path:
        out = Path(path)
        out.parent.mkdir(parents=True, exist_ok=True)
        doc = {
            "type": self.type_name,
            "version": 1,
            "payload": self.to_payload(),
        }
        out.write_text(json.dumps(doc, indent=2, sort_keys=True) + "\n")
        return out

    @classmethod
    def load(cls, path: str | Path) -> "RLModel":
        p = Path(path)
        doc = json.loads(p.read_text())

        type_name = doc.get("type")
        if not isinstance(type_name, str):
            raise ValueError("Model artifact missing string field 'type'")

        model_cls = _RL_MODEL_REGISTRY.get(type_name)
        if model_cls is None:
            known = ", ".join(sorted(_RL_MODEL_REGISTRY)) or "<none>"
            raise ValueError(f"Unknown RLModel type '{type_name}'. Known: {known}")

        payload = doc.get("payload")
        # Backward compatibility: older artifacts stored fields at top-level.
        if payload is None:
            payload = {k: v for k, v in doc.items() if k not in ("type", "version")}
        if not isinstance(payload, dict):
            raise ValueError("Model artifact field 'payload' must be an object")

        return model_cls.from_payload(payload)
