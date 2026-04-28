from __future__ import annotations

from dataclasses import dataclass, field

from .env_types import Task


@dataclass
class MorphologyConfig:
    """Brittle star morphology configuration.

    segments_per_arm defines the number of segments for each arm. The length of
    this list implicitly sets the number of arms. Use 0 segments to represent
    a fully amputated arm (e.g., [4, 0, 4, 2, 4] for a 5-arm morphology with
    arm 1 removed and arm 3 shortened).

    The upstream biorobot library natively supports per-arm segment counts.
    """

    segments_per_arm: list[int] = field(default_factory=lambda: [4, 4, 4, 4, 4])
    use_p_control: bool = True
    use_torque_control: bool = False

    @property
    def num_arms(self) -> int:
        return len(self.segments_per_arm)


@dataclass
class ArenaConfig:
    size: list[float] = field(default_factory=lambda: [10.0, 5.0])
    sand_ground_color: bool = True
    attach_target: bool = True
    wall_height: float = 1.5
    wall_thickness: float = 0.1


@dataclass
class EnvConfig:
    """Shared environment settings.

    Note: Some tasks have additional parameters (see fields below).
    """

    task: Task = Task.DIRECTED_LOCOMOTION

    simulation_time: float = 10000.0
    num_physics_steps_per_control_step: int = 10
    time_scale: int = 2

    camera_ids: list[int] = field(default_factory=lambda: [0, 1])
    # (height, width)
    render_size: list[int] = field(default_factory=lambda: [480, 640])

    joint_randomization_noise_scale: float = 0.0

    # Directed locomotion
    target_distance: float = 3.0

    # Light escape
    # Per docs in upstream env config: integer factors of 200.
    light_perlin_noise_scale: int = 0


@dataclass
class ObservationBoundsConfig:
    """Physical observation bounds for deterministic min-max normalization."""

    # TODO Inspect empirically observed ranges and update these bounds as needed.
    joint_position: list[float] = field(default_factory=lambda: [-3.14, 3.14])
    joint_velocity: list[float] = field(default_factory=lambda: [-20.0, 20.0])
    joint_actuator_force: list[float] = field(default_factory=lambda: [-5.0, 5.0])
    segment_contact: list[float] = field(default_factory=lambda: [0.0, 1.0])
    unit_xy_direction_to_target: list[float] = field(default_factory=lambda: [-1.0, 1.0])
    xy_distance_to_target: list[float] = field(default_factory=lambda: [0.0, 20.0])
    disk_z_tilt: list[float] = field(default_factory=lambda: [0.0, 3.141592653589793])

    def to_bounds_dict(self) -> dict[str, tuple[float, float]]:
        return {
            "joint_position": tuple(self.joint_position),
            "joint_velocity": tuple(self.joint_velocity),
            "joint_actuator_force": tuple(self.joint_actuator_force),
            "segment_contact": tuple(self.segment_contact),
            "unit_xy_direction_to_target": tuple(self.unit_xy_direction_to_target),
            "xy_distance_to_target": tuple(self.xy_distance_to_target),
            "disk_z_tilt": tuple(self.disk_z_tilt),
        }
