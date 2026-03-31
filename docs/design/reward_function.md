# Reward function and observation space

The robot needs to know whether its movements contribute to the ultimate goal of locomotion towards a target. Sensor
inputs must be distributed fairly to guarantee an objective comparison between different architectures.

- The distance from the robot to the target and/or the light intensity are treated as global inputs.
- Positions and joints, which are normalized to floating-point values between 0 and 1, are considered local inputs.
- The reward function is centered around minimizing the distance to the goal or maximizing the movement towards the goal
  within a finite number of timesteps $T$.
- To motivate efficient movement, the amount of timesteps taken to reach the goal will be used as penalty.

## Rationale

Using a light source (or a gradient) is biologically plausible for many simple organisms. By normalizing all signals
between 0 and 1, PPO training is highly stabilized. The timesteps must be finite to reset the environment in a timely
manner if the policy gets stuck in a local minimum.

## Limitations and alternatives

Providing global information to all individual decentralized segments can be considered biologically cheating or
practically infeasible once the robot would be physically built. Some sensory input cannot be put in each joint, for
example.

The alternative is to provide the global input to the outermost segments of the arms, or a specific set of segments
assigned with this functionality. The network would then have to learn to propagate this signal throughout the body via
message passing. While biologically more accurate, this drastically complicates the learning process. We have written
this down as potential future research.
