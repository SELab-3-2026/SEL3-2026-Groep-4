from dataclasses import dataclass, field
from typing import List, Optional


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

    name: str = "base"

    # Actor pipeline
    sensor: Optional[LayerConfig] = None
    propagator: Optional[LayerConfig] = None
    motor: Optional[LayerConfig] = None

    # Critic pipeline
    feature_extractor: Optional[LayerConfig] = None
    critic: Optional[LayerConfig] = None

    # Decentralized
    message_passing_steps: Optional[int] = None
    topology_type: Optional[str] = None  # Supported values: "ring", "fully_connected"


@dataclass
class CentralizedConfig(ArchitectureConfig):
    """Centralized actor-critic architecture (baseline).

    The actor is a single global policy composed of a sensor (input network)
    and a motor (output network). The sensor receives the full concatenated
    global observation; the motor projects the hidden state to all joint actions.

    See docs/design/actor-critic.md for the full design rationale.
    """

    name: str = "centralized"


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

    name: str = "decentralized"
