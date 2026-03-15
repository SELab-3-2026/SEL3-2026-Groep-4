from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Protocol

import numpy as np


@dataclass()
class SimulationConfig:
    duration_s: float = 10.0
    realtime: bool = True
    seed: int = 0


class ControlPolicy(Protocol):
    def act(self, *, obs: np.ndarray | None = None, t: float = 0.0) -> np.ndarray: ...


def _get_mjcf_xml(env: Any) -> str | None:
    for attr in ("mjcf", "_mjcf"):
        if hasattr(env, attr):
            mjcf_obj = getattr(env, attr)
            if hasattr(mjcf_obj, "get_mjcf_str"):
                return mjcf_obj.get_mjcf_str()

    for attr in ("get_mjcf_str", "mjcf_str", "mjcf_xml"):
        if hasattr(env, attr):
            v = getattr(env, attr)
            if callable(v):
                try:
                    return v()
                except TypeError:
                    pass
            elif isinstance(v, str):
                return v

    return None

def _get_model_data(env: Any):
    import mujoco

    candidates = [
        ("model", "data"),
        ("mj_model", "mj_data"),
        ("_model", "_data"),
    ]

    for model_attr, data_attr in candidates:
        if hasattr(env, model_attr) and hasattr(env, data_attr):
            model = getattr(env, model_attr)
            data = getattr(env, data_attr)
            if isinstance(model, mujoco.MjModel) and isinstance(data, mujoco.MjData):
                return model, data

    return None


def _get_model_data_from_state(state: Any):
    import mujoco

    candidates = [
        ("mj_model", "mj_data"),
        ("model", "data"),
        ("_model", "_data"),
    ]

    for model_attr, data_attr in candidates:
        if hasattr(state, model_attr) and hasattr(state, data_attr):
            model = getattr(state, model_attr)
            data = getattr(state, data_attr)
            if isinstance(model, mujoco.MjModel) and isinstance(data, mujoco.MjData):
                return model, data

    return None

def _default_obs_from_mjdata(data: Any) -> np.ndarray:
    qpos = np.asarray(data.qpos, dtype=np.float32).ravel()
    qvel = np.asarray(data.qvel, dtype=np.float32).ravel()
    return np.concatenate([qpos, qvel], axis=0)


def simulate_policy(
    env: Any,
    *,
    policy: ControlPolicy,
    config: SimulationConfig,
    state: Any | None = None,
) -> None:
    """Open MuJoCo's native viewer and step using actions from a policy.

    This path drives MuJoCo physics directly (mj_step) and uses the policy output
    as `data.ctrl`.
    """

    import mujoco
    import mujoco.viewer

    model_data = None
    if state is not None:
        model_data = _get_model_data_from_state(state)
    if model_data is None:
        model_data = _get_model_data(env)

    if model_data is not None:
        model, data = model_data
    else:
        xml = _get_mjcf_xml(env)
        if xml is None:
            raise RuntimeError(
                "Could not extract MuJoCo model/data from the provided state/env, or MJCF XML from the env. "
                "If this env is purely functional, pass the state returned by reset() or expose mjcf/model/data."
            )
        model = mujoco.MjModel.from_xml_string(xml)
        data = mujoco.MjData(model)

    start = time.time()
    with mujoco.viewer.launch_passive(model, data) as viewer:
        while viewer.is_running() and (time.time() - start) < config.duration_s:
            step_start = time.time()

            t = time.time() - start
            obs = _default_obs_from_mjdata(data)
            ctrl = policy.act(obs=obs, t=t)

            if model.nu > 0:
                ctrl = np.asarray(ctrl, dtype=np.float32).ravel()
                if ctrl.shape != (model.nu,):
                    raise ValueError(f"Policy returned ctrl shape {ctrl.shape}, expected ({model.nu},)")
                data.ctrl[:] = ctrl

            mujoco.mj_step(model, data)
            viewer.sync()

            if config.realtime:
                remaining = model.opt.timestep - (time.time() - step_start)
                if remaining > 0:
                    time.sleep(remaining)
