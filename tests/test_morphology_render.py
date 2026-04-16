import pytest
import os
import sys

# CRITICAL for headless cross-platform testing (devcontainers etc)
if sys.platform == "linux" and "DISPLAY" not in os.environ and "WAYLAND_DISPLAY" not in os.environ:
    os.environ.setdefault("MUJOCO_GL", "egl")

import mujoco
from PIL import Image
from brittle_star_project.environment.env_config import EnvConfig, MorphologyConfig, ArenaConfig
from brittle_star_project.environment.BrittleStarJaxEnvWrapper import BrittleStarJaxEnvWrapper


@pytest.mark.skipif(os.getenv("CI") == "true", reason="No OpenGL display in CI")
def test_render_morphologies():
    base_dir = "runs/renders"
    os.makedirs(base_dir, exist_ok=True)

    # --- 1. Full 5-Arm Morphology ---
    morph_full = MorphologyConfig(segments_per_arm=[4, 4, 4, 4, 4])
    env_full = BrittleStarJaxEnvWrapper(
        morphology=morph_full, arena=ArenaConfig(), env_config=EnvConfig(), num_envs=1
    )
    state_full = env_full.reset(seed=0)

    model_full = state_full.mj_model
    data_full = state_full.mj_data

    # 1. Compute forward kinematics so geoms are correctly positioned
    mujoco.mj_forward(model_full, data_full)

    # 2. Render using the environment's primary camera (camera=0)
    renderer_full = mujoco.Renderer(model=model_full)
    renderer_full.update_scene(data_full, camera=1)
    pixels_full = renderer_full.render()
    image_path = os.path.join(base_dir, "full_5_arm.png")
    Image.fromarray(pixels_full).save(image_path)
    print(f"Generated full morphology render: {image_path}")

    # --- 2. Partially Amputated Morphology ---
    morph_amp = MorphologyConfig(segments_per_arm=[4, 0, 4, 2, 4])
    env_amp = BrittleStarJaxEnvWrapper(
        morphology=morph_amp, arena=ArenaConfig(), env_config=EnvConfig(), num_envs=1
    )
    state_amp = env_amp.reset(seed=0)

    model_amp = state_amp.mj_model
    data_amp = state_amp.mj_data

    # Compute forward kinematics
    mujoco.mj_forward(model_amp, data_amp)

    renderer_amp = mujoco.Renderer(model=model_amp)
    renderer_amp.update_scene(data_amp, camera=1)
    pixels_amp = renderer_amp.render()
    image_path = os.path.join(base_dir, "amputated_arm.png")
    Image.fromarray(pixels_amp).save(image_path)
    print(f"Generated amputated morphology render: {image_path}")

    print("Morphology render test successful!")


if __name__ == "__main__":
    test_render_morphologies()
