from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class LayerConfig:
    hidden_dims: List[int] = field(default_factory=lambda: [64, 64])
    activation: str = "relu"


@dataclass
class MLPConfig:
    actor: LayerConfig = field(default_factory=LayerConfig)
    critic: LayerConfig = field(default_factory=LayerConfig)
    sensor: LayerConfig = field(default_factory=LayerConfig)
    feature_extractor: LayerConfig = field(default_factory=LayerConfig)


@dataclass
class NetworksConfig:
    mlp: MLPConfig = field(default_factory=MLPConfig)
    # Future placeholder for message_passing
    # message_passing: Optional[MessagePassingConfig] = None
