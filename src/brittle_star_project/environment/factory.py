from __future__ import annotations

from dataclasses import asdict

from .env_config import ArenaConfig, EnvConfig, MorphologyConfig
from .env_types import Backend, Task


class BrittleStarEnvFactory:
    """Creates brittle-star morphology, arena, and task environment instances."""

    def __init__(self, backend: Backend) -> None:
        self._backend = backend

    @property
    def backend(self) -> Backend:
        return self._backend

    def create_morphology(self, config: MorphologyConfig):
        from biorobot.brittle_star.mjcf.morphology.morphology import (
            MJCFBrittleStarMorphology,
        )
        from biorobot.brittle_star.mjcf.morphology.specification.default import (
            default_brittle_star_morphology_specification,
        )

        spec = default_brittle_star_morphology_specification(
            num_arms=config.num_arms,
            num_segments_per_arm=config.num_segments_per_arm,
            use_p_control=config.use_p_control,
            use_torque_control=config.use_torque_control,
        )
        return MJCFBrittleStarMorphology(specification=spec)

    def create_arena(self, config: ArenaConfig):
        from biorobot.brittle_star.mjcf.arena.aquarium import (
            AquariumArenaConfiguration,
            MJCFAquariumArena,
        )

        arena_config = AquariumArenaConfiguration(**asdict(config))
        return MJCFAquariumArena(configuration=arena_config)

    def create_environment_configuration(self, config: EnvConfig):
        # Import locally so the project can still be imported without these deps.
        from biorobot.brittle_star.environment.directed_locomotion.shared import (
            BrittleStarDirectedLocomotionEnvironmentConfiguration,
        )
        from biorobot.brittle_star.environment.light_escape.shared import (
            BrittleStarLightEscapeEnvironmentConfiguration,
        )
        from biorobot.brittle_star.environment.undirected_locomotion.shared import (
            BrittleStarUndirectedLocomotionEnvironmentConfiguration,
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

        # TODO: replace with match statement?
        if config.task == Task.DIRECTED_LOCOMOTION:
            return BrittleStarDirectedLocomotionEnvironmentConfiguration(
                target_distance=config.target_distance,
                **common,
            )

        if config.task == Task.LIGHT_ESCAPE:
            return BrittleStarLightEscapeEnvironmentConfiguration(
                light_perlin_noise_scale=config.light_perlin_noise_scale,
                **common,
            )

        raise ValueError(f"Unsupported task: {config.task}")

    def create_environment(self, morphology_config: MorphologyConfig, arena_config: ArenaConfig, env_config: EnvConfig):
        from biorobot.brittle_star.environment.directed_locomotion.dual import (
            BrittleStarDirectedLocomotionEnvironment,
        )
        from biorobot.brittle_star.environment.light_escape.dual import (
            BrittleStarLightEscapeEnvironment,
        )

        morphology = self.create_morphology(morphology_config)
        arena = self.create_arena(arena_config)
        env_configuration = self.create_environment_configuration(env_config)

        if env_config.task == Task.DIRECTED_LOCOMOTION:
            env_class = BrittleStarDirectedLocomotionEnvironment
        elif env_config.task == Task.LIGHT_ESCAPE:
            env_class = BrittleStarLightEscapeEnvironment
        else:
            raise ValueError(f"Unsupported task: {env_config.task}")

        return env_class.from_morphology_and_arena(
            morphology=morphology,
            arena=arena,
            configuration=env_configuration,
            backend=self._backend.value,
        )
