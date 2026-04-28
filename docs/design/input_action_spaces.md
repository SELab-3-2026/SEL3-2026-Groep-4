# Input (state) and output (action) spaces

To effectively learn locomotion and navigation, the agent requires a well-defined observation space (inputs) and action
space (outputs). The control models map these observations directly to physical movements.

**Inputs (state space)**

The observation space provides the agent with its current physical state and its navigational objective. With a
decentralized control architecture in mind, we divide these inputs into global and local states.

Global inputs, always broadcasted to all nodes:

- Vertical orientation/tilt: A single, simplified metric representing the tilt/vertical alignment of the agent's
  central body/disk, a.k.a. the deviation from the global Z-axis. Its value is derived from the environment's raw disk
  rotation 3D vector $[roll, pitch, yaw]$:
  $$
  tilt = sqrt(roll^2 + pitch^2)
  $$
- Goal vector: Instead of just a scalar distance, the goal is represented asa a vector (distance and ange/direction) to
  the target.

Local inputs, routed directly to specific nodes:

- Joint positions: The current angles of all joints within the morphology.
- Joint velocities: The current angular velocities of the joints.
- Joint actuator forces: The physical forces currently exerted at each specific joint.
- Segment contact: These values indicate whether each physical segment of the agent is currently touching the ground.

**Outputs (action space)**

The action space defines how the agent interacts with the environment.

- Joint offsets: *absolute* target positions (offsets) for the joints, i.e. the exact angle the joint should move to.

## Normalization and Scaling

Both the input (observation) and output (action) spaces are rescaled to the range **$[-1, 1]$**. 

For the input space, all raw physical values (angles, velocities, forces, distances) are normalized based on their
defined physical bounds. If a value exceeds these bounds during simulation, it is clipped to the $[-1, 1]$ range.

For the output space, the neural network's tanh-activated outputs (which naturally fall in $[-1, 1]$) are linearly
mapped to the physical joint limits defined in the robot's morphology.

## Rationale

When designing the state space, we must ask: *Could a human operator perform this task given only these inputs?*

- Inclusion of Joint Velocities: Because our control models do not inherently possess memory of previous timesteps,
  providing only the joint position is insufficient to determine the direction a limb is currently moving. By
  explicitly including joint velocities, the agent can immediately infer momentum and movement direction without
  needing to memorize past states.
- Absolute Joint Offsets: The physical Brittle Star robot relies on servo motors (if we were to build this simulated
  robot), which are inherently position-controlled devices. (Continuous rotation servos exist, but they are less
  commonly used for joints.) If our network outputted continuous torques (forces), a significant portion of the
  reinforcement learning process would be wasted on learning low-level PID control dynamics (i.e., how much force to
  apply to hold a position). Abstracting this away forces the learning algorithm to focus entirely on higher-level gait
  generation and locomotion.
- Simplified vertical orientation: We drop the full 3D spatial rotation and angular velocity arrays in favor of a
  single vertical orientation metric (tilt). For a brittle star moving accross a flat plane, this metric is sufficient
  for the agent to sense if it is losing balance or flipping over.
- Force representation: We strictly retain the joint actuator forces and drop the more generic actuator force. Forces
  that are explicitly tied to individual joints are significantly easier to route into decentralized, local limb nodes,
  which is necessary for our message-passing architecture.
- Goal Vector (Distance + Angle): Providing only the scalar "distance to the goal" as an input is akin to blindfolding
  the robot and asking it to find a target by playing "hot or cold." By providing a full vector, the agent knows
  exactly where the target is relative to its current orientation, allowing for directed and efficient locomotion.

  The goal representation is explicitly divided into a directional vector and a scalar distance. Keeping the direction
  as a normalized unit vector bounds the values to the $[-1, 1]$ range, which stabilizes neural network training.
  Providing only a scalar "distance to the goal" would force the agent to learning localized searching behaviors (e.g.
  random walks or spiraling) to deduce the direction, drastically increasing the difficulty of the learning task.
- Contact sensors: Segment contact detects external ground interaction and is biologically vital for timing gait
  transitions.
- **Zero-Centered Rescaling ($[-1, 1]$):** Using a zero-centered range is standard best practice for continuous control
  tasks. It provides several mathematical and physical advantages:
  - **Improved Gradient Flow:** Neural networks optimize faster when inputs are zero-centered. If all inputs were
    positive (e.g., $[0, 1]$), the gradients during backpropagation would be forced to the same sign, causing
    inefficient "zig-zag" weight updates.
  - **Meaningful "Neutral" State:** In robotics, $0.0$ naturally represents a resting state (zero velocity, centered
    position, no force). In a $[-1, 1]$ system, this physical rest maps to a neutral $0.0$ signal in the network.
  - **Robustness to Amputation:** In this project, amputated limbs are padded with $0.0$. In a $[-1, 1]$ system, this
    correctly communicates a "neutral/dead" signal. In a $[0, 1]$ system, $0.0$ would represent the absolute minimum
    physical limit, causing the network to misinterpret missing limbs as being at their extreme limits.



Specifically, we do not include some available inputs:

- Global position: Absolute spatial coordinates can cause the agent to overfit to a specific coordinate frame or map,
  rather than learning general, adaptable locomotion strategies.

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
- **$[0, 1]$ Rescaling:** While some domains (like computer vision) use $[0, 1]$ scaling, it is generally avoided in
  robotics. Scaling to $[0, 1]$ would mean that a resting joint (velocity = 0) maps to an input of $0.5$. This
  constant positive bias forces the network to waste capacity learning to ignore or subtract this baseline signal just
  to stand still. Furthermore, it breaks the "dead signal" interpretation of zero-padding used for amputations.

## MuJoCo

This is what the filtered input vectors look like in MuJoCo, with $J$ joints and $S$ segments:

- `joint_position`: shape=(J,), dtype=float64
- `joint_velocity`: shape=(J,), dtype=float64
- `joint_actuator_force`: shape=(J,), dtype=float64
- `segment_contact`: shape=(S,), dtype=float64
- `unit_xy_direction_to_target`: shape=(2,), dtype=float64
- `xy_distance_to_target`: shape=(1,), dtype=float64
- `disk_z_tilt`: shape=(1,), dtype=float64, derived from `disk_rotation`

This brings the entire input space down to $3J + S + 4$ float64's, compared to $4J + S + 15$ float64's for the
unfiltered inputs.

For reference, these are all the inputs that are available in the MuJoCo environment:

```
obs keys: ['joint_position', 'joint_velocity', 'joint_actuator_force', 'actuator_force', 'disk_position', 'disk_rotation', 'disk_linear_velocity', 'disk_angular_velocity', 'tendon_position', 'tendon_velocity', 'segment_contact', 'unit_xy_direction_to_target', 'xy_distance_to_target']

raw observations dict:
{'joint_position': array([0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.]),
 'joint_velocity': array([0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.]),
 'joint_actuator_force': array([0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.]),
 'actuator_force': array([0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.]),
 'disk_position': array([0.  , 0.  , 0.11]),
 'disk_rotation': (0.0, -0.0, 0.0),
 'disk_linear_velocity': array([0., 0., 0.]),
 'disk_angular_velocity': array([0., 0., 0.]),
 'tendon_position': array([], dtype=float64),
 'tendon_velocity': array([], dtype=float64),
 'segment_contact': array([0., 0., 0., 0., 0., 0.]),
 'unit_xy_direction_to_target': array([-0.95333378, -0.30191837]),
 'xy_distance_to_target': array([3.])}

(shapes)
joint_position: shape=(12,), dtype=float64, size=12
joint_velocity: shape=(12,), dtype=float64, size=12
joint_actuator_force: shape=(12,), dtype=float64, size=12
actuator_force: shape=(12,), dtype=float64, size=12
disk_position: shape=(3,), dtype=float64, size=3
disk_rotation: shape=(3,), dtype=float64, size=3
disk_linear_velocity: shape=(3,), dtype=float64, size=3
disk_angular_velocity: shape=(3,), dtype=float64, size=3
tendon_position: shape=(0,), dtype=float64, size=0
tendon_velocity: shape=(0,), dtype=float64, size=0
segment_contact: shape=(6,), dtype=float64, size=6
unit_xy_direction_to_target: shape=(2,), dtype=float64, size=2
xy_distance_to_target: shape=(1,), dtype=float64, size=1
```
