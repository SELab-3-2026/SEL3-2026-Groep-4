from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Protocol

import numpy as np


@dataclass
class SimulationConfig:
    realtime: bool = True
    seed: int = 0


class ControlPolicy(Protocol):
    def act(self, *, obs: np.ndarray | None = None, t: float = 0.0) -> np.ndarray: ...


def _default_observations(data: Any) -> np.ndarray:
    qpos = np.asarray(data.qpos, dtype=np.float32).ravel()
    qvel = np.asarray(data.qvel, dtype=np.float32).ravel()
    return np.concatenate([qpos, qvel], axis=0)


def simulate_policy(
    policy: ControlPolicy,
    config: SimulationConfig,
    state: Any | None = None,
) -> None:
    """Open MuJoCo's native viewer and step using actions from a policy.

    This path drives MuJoCo physics directly (mj_step) and uses the policy output
    as `data.ctrl`.
    """

    import mujoco.viewer

    model = state.mj_model
    data = state.mj_data

    start = time.time()
    with mujoco.viewer.launch_passive(model, data) as viewer:
        while viewer.is_running():
            step_start = time.time()

            t = time.time() - start

            # Input vector for the policy
            # TODO: custom input
            obs = _default_observations(data)

            # Policy action
            ctrl = policy.act(obs=obs, t=t)

            # Check if the policy output vector give an input for each actuator (nu)
            # TODO: what if model trained on full morphology but we want to test on a damaged one? (nu mismatch)
            if model.nu > 0:
                ctrl = np.asarray(ctrl, dtype=np.float32).ravel()
                if ctrl.shape != (model.nu,):
                    raise ValueError(
                        f"Policy returned ctrl shape {ctrl.shape}, expected ({model.nu},)"
                    )
                data.ctrl[:] = ctrl

            # Step the simulation and update the viewer
            mujoco.mj_step(model, data)
            viewer.sync()

            # If we're running in realtime mode, sleep to maintain real-time pacing.
            if config.realtime:
                remaining = model.opt.timestep - (time.time() - step_start)
                if remaining > 0:
                    time.sleep(remaining)
