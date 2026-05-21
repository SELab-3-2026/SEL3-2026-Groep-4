from __future__ import annotations

from dataclasses import asdict

from moojoco.environment.dual import DualMuJoCoEnvironment

from .env_config import ArenaConfig, EnvConfig, MorphologyConfig
from .env_types import Backend, Task


class BrittleStarEnvFactory:
    """Creates brittle-star morphology, arena, and task environment instances."""

    @staticmethod
    def create_morphology(config: MorphologyConfig):
        from biorobot.brittle_star.mjcf.morphology.morphology import (
            MJCFBrittleStarMorphology,
        )
        from biorobot.brittle_star.mjcf.morphology.specification.default import (
            default_brittle_star_morphology_specification,
        )

        spec = default_brittle_star_morphology_specification(
            num_arms=config.num_arms,
            num_segments_per_arm=list(config.segments_per_arm),
            use_p_control=config.use_p_control,
            use_torque_control=config.use_torque_control,
        )
        return MJCFBrittleStarMorphology(specification=spec)

    @staticmethod
    def create_arena(config: ArenaConfig):
        from biorobot.brittle_star.mjcf.arena.aquarium import (
            AquariumArenaConfiguration,
            MJCFAquariumArena,
        )

        arena_config = AquariumArenaConfiguration(**asdict(config))
        return MJCFAquariumArena(configuration=arena_config)

    @staticmethod
    def create_environment_configuration(config: EnvConfig):
        # Import locally so the project can still be imported without these deps.
        from biorobot.brittle_star.environment.directed_locomotion.shared import (
            BrittleStarDirectedLocomotionEnvironmentConfiguration,
        )
        from biorobot.brittle_star.environment.light_escape.shared import (
            BrittleStarLightEscapeEnvironmentConfiguration,
        )

        common = dict(
            joint_randomization_noise_scale=config.joint_randomization_noise_scale,
            render_mode="human",
            simulation_time=config.simulation_time,
            num_physics_steps_per_control_step=config.num_physics_steps_per_control_step,
            time_scale=config.time_scale,
            camera_ids=config.camera_ids,
            render_size=config.render_size,
        )

        match config.task:
            case Task.DIRECTED_LOCOMOTION:
                return BrittleStarDirectedLocomotionEnvironmentConfiguration(
                    target_distance=config.target_distance,
                    **common,
                )
            case Task.LIGHT_ESCAPE:
                return BrittleStarLightEscapeEnvironmentConfiguration(
                    light_perlin_noise_scale=config.light_perlin_noise_scale,
                    **common,
                )
            case _:
                raise ValueError(f"Unsupported task: {config.task}")

    @staticmethod
    def create_environment(
        backend: Backend,
        morphology_config: MorphologyConfig,
        arena_config: ArenaConfig,
        env_config: EnvConfig,
    ) -> DualMuJoCoEnvironment:
        from biorobot.brittle_star.environment.directed_locomotion.dual import (
            BrittleStarDirectedLocomotionEnvironment,
        )
        from biorobot.brittle_star.environment.light_escape.dual import (
            BrittleStarLightEscapeEnvironment,
        )

        morphology = BrittleStarEnvFactory.create_morphology(morphology_config)
        arena = BrittleStarEnvFactory.create_arena(arena_config)
        env_configuration = BrittleStarEnvFactory.create_environment_configuration(env_config)

        match env_config.task:
            case Task.DIRECTED_LOCOMOTION:
                env_class = BrittleStarDirectedLocomotionEnvironment
            case Task.LIGHT_ESCAPE:
                env_class = BrittleStarLightEscapeEnvironment
            case _:
                raise ValueError(f"Unsupported task: {env_config.task}")

        env = env_class.from_morphology_and_arena(
            morphology=morphology,
            arena=arena,
            configuration=env_configuration,
            backend=backend.value,
        )

        from experiment_logger import get_logger

        get_logger().info(f"Created {env_config.task.value} env on backend {backend.value}")

        return env
