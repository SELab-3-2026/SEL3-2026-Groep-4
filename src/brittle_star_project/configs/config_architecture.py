from dataclasses import dataclass, field
from typing import List


@dataclass
class LayerConfig:
    hidden_dims: List[int] = field(default_factory=lambda: [64, 64])
    activation: str = "tanh"


@dataclass
class ArchitectureConfig:
    """Base class for actor-critic network configurations.

    Both centralized and decentralized architectures share a centralized critic
    composed of a feature extractor followed by a shallow output layer.

    See docs/design/actor-critic.md for the full design rationale.
    """

    # Input network: maps global observation to a hidden representation.
    feature_extractor: LayerConfig = field(
        default_factory=lambda: LayerConfig(hidden_dims=[64, 64], activation="tanh")
    )
    # Output network: maps the hidden representation to a scalar value estimate.
    critic: LayerConfig = field(
        default_factory=lambda: LayerConfig(hidden_dims=[], activation="tanh")
    )


@dataclass
class CentralizedConfig(ArchitectureConfig):
    """Centralized actor-critic architecture (baseline).

    The actor is a single global policy composed of a sensor (input network)
    and a motor (output network). The sensor receives the full concatenated
    global observation; the motor projects the hidden state to all joint actions.

    See docs/design/actor-critic.md for the full design rationale.
    """

    # Input network: maps the global observation to a hidden state.
    sensor: LayerConfig = field(
        default_factory=lambda: LayerConfig(hidden_dims=[64, 64], activation="tanh")
    )
    # Output network: projects hidden state to (mean, log_std) over all joints.
    motor: LayerConfig = field(
        default_factory=lambda: LayerConfig(hidden_dims=[], activation="tanh")
    )


@dataclass
class DecentralizedConfig(ArchitectureConfig):
    """Decentralized actor architecture (NerveNet-MLP variant).

    Each node runs a local sensor, exchanges messages with neighbours via a
    propagator for a fixed number of steps, and then a local motor produces
    the joint offset for that node only.

    The critic remains centralized (shared with the base class): it receives the
    full concatenated global observation and outputs a single scalar.

    See docs/design/actor-critic.md and docs/design/communication.md for the
    full design rationale.
    """

    # Input network: local observation -> initial hidden state per node.
    sensor: LayerConfig = field(
        default_factory=lambda: LayerConfig(hidden_dims=[64, 64], activation="tanh")
    )
    # Message-passing network: aggregates neighbour messages and updates hidden state.
    propagator: LayerConfig = field(
        default_factory=lambda: LayerConfig(hidden_dims=[64, 64], activation="tanh")
    )
    # Output network: final hidden state -> joint offset for this node only.
    motor: LayerConfig = field(
        default_factory=lambda: LayerConfig(hidden_dims=[], activation="tanh")
    )

    # Number of synchronous message-passing rounds per control step.
    message_passing_steps: int = 1
    # Graph topology used for neighbour connections.
    # Supported values: "ring", "fully_connected".
    topology_type: str = "ring"
