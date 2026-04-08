# Input (state) and output (action) spaces

To effectively learn locomotion and navigation, the agent requires a well-defined observation space (inputs) and action
space (outputs). The control models map these observations directly to physical movements.

**Inputs (state space)**

The observation space provides the agent with its current physical state and its objective.

- Joint positions: the current angles of all joints in the morphology.
- Joint velocities: the current moving speed of the joints.
- Goal vector: instad of just a scalar distance, the goal is represented asa a vector (distance and ange/direction) to
  the target.

**Outputs (action space)**

The action space defines how the agent interacts with the environment.

- Joint offsets: *absolute* target positions (offsets) for the joints, i.e. the exact angle the joint should move to.

## Rationale

When designing the state space, we must ask: *Could a human operator perform this task given only these inputs?*

- Inclusion of Joint Velocities: Because our control models do not inherently possess memory of previous timesteps,
  providing only the joint position is insufficient to determine the direction a limb is currently moving. By
  explicitly including joint velocities, the agent can immediately infer momentum and movement direction without
  needing to memorize past states.
- Goal Vector (Distance + Angle): Providing only the scalar "distance to the goal" as an input is akin to blindfolding
  the robot and asking it to find a target by playing "hot or cold." By providing a full vector, the agent knows
  exactly where the target is relative to its current orientation, allowing for directed and efficient locomotion.
- Absolute Joint Offsets: The physical Brittle Star robot relies on servo motors (if we were to build this simulated
  robot), which are inherently position-controlled devices. (Continuous rotation servos exist, but they are less
  commonly used for joints.) If our network outputted continuous torques (forces), a significant portion of the
  reinforcement learning process would be wasted on learning low-level PID control dynamics (i.e., how much force to
  apply to hold a position). Abstracting this away forces the learning algorithm to focus entirely on higher-level gait
  generation and locomotion.

## Limitations and alternatives

Alternative state and action formulations include:

- Torque-based continuous control: In many continuous control tasks (like standard MuJoCo benchmarks), actions
  represent continuous torques applied to joints. While this provides more granular, low-level physical control, it
  heavily complicates training and does not align well with the physical reality of servo-driven hardware.
- Recurrent Neural Networks (RNNs) / Frame Stacking: Instead of explicitly passing velocities in the state space, the
  network could infer momentum by observing a history of past states. Using RNNs or frame stacking allows the agent to
  build an internal memory of movement. However, this significantly increases architectural complexity and training
  time compared to explicitly providing the velocity data.
- Scalar Goal Distance: Giving the agent only the scalar distance to the target would force it to learn a localized
  searching behavior (e.g., spiraling or random walks) to determine the correct direction. While biologically plausible
  for simpler organisms following chemical gradients, it drastically increases the difficulty of the learning task.
