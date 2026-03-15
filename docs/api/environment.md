# Brittle star environment

## Creation
The environment package contains a factory class `BrittleStarEnvFactory` 
that creates instances of the environment/morphologies/... It uses the
configuration classes defined in `env_config.py` to create the instances.

## Configuration
The data classes in `env_config` have default values as stated in the tutorials.
* MorphologyConfig: configuration for the morphology of the brittle star. Contains
number of arms, number of segments per arm, and control mode.
* ArenaConfig: configuration for the arena. Sets the size of the arena, whether to 
set the ground floor to sand, attach a target and sizes of the walls.
* EnvConfig: configuration for the environment. These set shared settings
such as camera locations, simulation time and the task.

## Backend and Task enums
The Backend enum specifies either an MJC or MJX backend.
* MJC: runs on CPU
* MJX: uses jax on the gpu

The Task enum specifies which task to use. 3 items are present:
* DIRECTED_LOCOMOTION: move to a target location
* LIGHT_ESCAPE: situation where the robot must move to a darker location
