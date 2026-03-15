from __future__ import annotations

from enum import Enum


class Backend(str, Enum):
    """Physics backend.

    - MJC: MuJoCo C engine
    - MJX: MuJoCo XLA (JAX) engine
    """

    MJC = "MJC"
    MJX = "MJX"


class Task(str, Enum):
    """Which brittle-star task/environment to instantiate."""

    UNDIRECTED_LOCOMOTION = "undirected_locomotion"
    DIRECTED_LOCOMOTION = "directed_locomotion"
    LIGHT_ESCAPE = "light_escape"
