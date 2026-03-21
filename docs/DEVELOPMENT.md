# Development Guide

This guide outlines how to set up the development environment for this project, prioritizing **reproducible builds**, **environment parity**, and **cross-hardware compatibility**.

## Reproducibility &uv

This project uses [uv](https://github.com/astral-sh/uv) to manage dependencies and virtual environments. The `uv.lock` file is the absolute source of truth for package versions and must always be committed.

### Source of Truth

- **Never modify `uv.lock` manually.**
- To add a dependency, run `uv add <package>`.
- To update dependencies, run `uv lock --upgrade`.
- To sync your environment with the lockfile, run `uv sync --frozen`.

## Git LFS (Critical)

**All developers must have Git LFS installed locally.** This repository tracks model weights (`.pt`, `.safetensors`, etc.), recordings (`.mp4`), and datasets using Git LFS.

- **Setup:** Run `git lfs install` after cloning this repository. If you are using the `.devcontainer` or `flake.nix`, LFS is typically available automatically.
- If you clone without LFS installed, run `git lfs pull` after installation to fetch the actual data files instead of the small pointer files.

## Devcontainer Setup (Recommended)

The devcontainer provides an identical experience to local development but with all system dependencies pre-configured. We provide three specialized configurations to match your hardware.

### Prerequisites

- Docker Desktop or Docker Engine.
- **For NVIDIA GPUs:** [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html).
- **For AMD GPUs:** A host with ROCm installed and appropriate permissions (usually `video` and `render` groups).

### Setup for VS Code

1. Install the [Dev Containers](https://marketplace.visualstudio.com/items?itemName=ms-vscode-remote.remote-containers) extension.
2. Open the project and click **Reopen in Container**.
3. VS Code will prompt you to select a configuration:
   - **Brittle Star JAX/CUDA**: Choose this if you have an NVIDIA GPU.
   - **Brittle Star JAX/ROCm**: Choose this if you have an AMD GPU.
   - **Brittle Star JAX/CPU**: Choose this for a standard CPU environment (e.g., Mac, laptops).
4. On first launch, the `post-create.sh` script will automatically sync the correct hardware-specific dependencies using `uv`.
5. The environment is stored in a **named volume** for `.venv` to ensure persistence and performance.

### Setup for JetBrains IDEs

1. The IDE will detect the configurations in the `.devcontainer` directory.
2. Select your preferred configuration (CUDA, ROCm, or CPU).
3. The hardware-aware sync will run automatically during container creation.

## Local Development (Alternative)

If you prefer not to use Docker:
1. Install [uv](https://docs.astral.sh/uv/getting-started/installation/).
2. Run the appropriate sync command:
   - **CPU:** `uv sync --frozen`
   - **NVIDIA GPU:** `uv sync --frozen --extra cuda`
   - **AMD GPU:** `uv sync --frozen --extra rocm`

## Hardware Acceleration (JAX)

Verify your setup by running the JAX initialization test:

```bash
uv run pytest tests/test_jax_init.py
```
In the devcontainer, this will succeed on CPU, NVIDIA GPU, or AMD GPU depending on your chosen configuration. A `GpuDevice` is expected if a GPU is detected and the appropriate extra was installed.
