from __future__ import annotations

from dataclasses import dataclass, field

from .env_types import Task


@dataclass(frozen=True, slots=True)
class MorphologyConfig:
    num_arms: int = 5
    num_segments_per_arm: int = 4
    use_p_control: bool = True
    use_torque_control: bool = False


@dataclass(frozen=True, slots=True)
class ArenaConfig:
    size: tuple[float, float] = (10.0, 5.0)
    sand_ground_color: bool = True
    attach_target: bool = True
    wall_height: float = 1.5
    wall_thickness: float = 0.1


@dataclass(frozen=True, slots=True)
class EnvConfig:
    """Shared environment settings.

    Note: Some tasks have additional parameters (see fields below).
    """

    task: Task = Task.DIRECTED_LOCOMOTION

    simulation_time: float = 5.0
    num_physics_steps_per_control_step: int = 10
    time_scale: int = 2

    camera_ids: list[int] = field(default_factory=lambda: [0, 1])
    # (height, width)
    render_size: tuple[int, int] = (480, 640)

    joint_randomization_noise_scale: float = 0.0

    # Directed locomotion
    target_distance: float = 3.0

    # Light escape
    # Per docs in upstream env config: integer factors of 200.
    light_perlin_noise_scale: int = 0


def from_file(path: str) -> tuple[MorphologyConfig, ArenaConfig, EnvConfig]:
    """Load configurations from a YAML file."""
    import yaml

    with open(path, "r") as f:
        config_dict = yaml.safe_load(f)

        morphology = MorphologyConfig(**config_dict.get("morphology", {}))
        arena = ArenaConfig(**config_dict.get("arena", {}))
        env = EnvConfig(**config_dict.get("env", {}))
        return morphology, arena, env
