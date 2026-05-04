from __future__ import annotations

import yaml
from dataclasses import dataclass
from pathlib import Path

import flax
from omegaconf import OmegaConf

from brittle_star_project.environment.env_config import (
    MorphologyConfig,
    ArenaConfig,
    EnvConfig,
    ObservationBoundsConfig,
)


@dataclass
class TrainingConfig:
    """Holds typed configurations extracted from a training run's metadata."""

    morphology: MorphologyConfig
    arena: ArenaConfig
    environment: EnvConfig
    obs_bounds: ObservationBoundsConfig


def load_params(path: Path) -> dict:
    """Load model parameters from a .flax checkpoint file."""
    payload = path.read_bytes()
    restored = flax.serialization.msgpack_restore(payload)

    sensor_params = None
    actor_params = None

    # Extract params from restored checkpoint
    if isinstance(restored, dict):
        params_sub = restored.get("params", {})
        sensor_params = restored.get("sensor_params") or params_sub.get("sensor_params")
        actor_params = restored.get("actor_params") or params_sub.get("actor_params")
    elif isinstance(restored, (list, tuple)) and len(restored) >= 2:
        params_part = restored[1]
        if isinstance(params_part, dict):
            sensor_params = params_part.get("0", params_part.get(0))
            actor_params = params_part.get("1", params_part.get(1))
        elif isinstance(params_part, (list, tuple)) and len(params_part) >= 2:
            sensor_params = params_part[0]
            actor_params = params_part[1]

    if sensor_params is None or actor_params is None:
        raise ValueError(f"Could not extract sensor and actor params from checkpoint: {path}")

    return {
        "sensor_params": sensor_params,
        "actor_params": actor_params,
    }


def load_metadata(model_path: Path, metadata_override_path: Path | None = None) -> dict:
    """Discover and load the sidecar metadata YAML file."""
    if metadata_override_path is not None:
        metadata_path = metadata_override_path
    else:
        metadata_path = model_path.with_name(model_path.stem + "_metadata.yaml")

    if not metadata_path.exists():
        raise FileNotFoundError(f"Could not find metadata YAML at {metadata_path}")
    with open(metadata_path, "r") as f:
        return yaml.safe_load(f)


def metadata_to_configs(metadata: dict) -> TrainingConfig:
    """Reconstruct typed configuration objects from a metadata dictionary."""
    trained_morphology = OmegaConf.to_object(
        OmegaConf.merge(OmegaConf.structured(MorphologyConfig), metadata.get("morphology", {}))
    )
    trained_arena = OmegaConf.to_object(
        OmegaConf.merge(OmegaConf.structured(ArenaConfig), metadata.get("arena", {}))
    )

    env_dict = metadata.get("environment", {})
    if isinstance(env_dict.get("task"), str):
        from brittle_star_project.environment.env_types import Task

        try:
            env_dict["task"] = Task[env_dict["task"]].name
        except Exception:
            try:
                env_dict["task"] = Task(env_dict["task"]).name
            except Exception:
                pass

    trained_environment = OmegaConf.to_object(
        OmegaConf.merge(OmegaConf.structured(EnvConfig), env_dict)
    )
    trained_obs_bounds = OmegaConf.to_object(
        OmegaConf.merge(
            OmegaConf.structured(ObservationBoundsConfig), metadata.get("obs_bounds", {})
        )
    )

    return TrainingConfig(
        morphology=trained_morphology,
        arena=trained_arena,
        environment=trained_environment,
        obs_bounds=trained_obs_bounds,
    )
