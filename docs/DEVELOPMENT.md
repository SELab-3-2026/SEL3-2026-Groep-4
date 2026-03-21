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

The devcontainer provides an identical experience to local development but with all system dependencies pre-configured. It automatically detects your hardware (GPU vs CPU) and syncs the appropriate dependencies.

### Prerequisites

- Docker Desktop or Docker Engine.
- [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html) (for GPU support).

### Setup for VS Code

1. Install the [Dev Containers](https://marketplace.visualstudio.com/items?itemName=ms-vscode-remote.remote-containers) extension.
2. Open the project and click **Reopen in Container**.
3. On first launch, the `post-create.sh` script will:
   - Detect if an NVIDIA GPU is available via `nvidia-smi`.
   - Run `uv sync --frozen --extra cuda` if a GPU is found.
   - Run `uv sync --frozen` otherwise.
4. The environment is stored in a **named volume** for `.venv` to ensure persistence and performance.

### Setup for JetBrains IDEs

1. The IDE will detect the `.devcontainer/devcontainer.json` file.
2. The environment is pre-configured to point to `/workspaces/project/.venv`.
3. The hardware-aware sync will run automatically during container creation.

## Local Development (Alternative)

If you prefer not to use Docker:
1. Install [uv](https://docs.astral.sh/uv/getting-started/installation/).
2. Run `uv sync --frozen` (CPU) or `uv sync --frozen --extra cuda` (GPU).

## Hardware Acceleration (JAX)

Verify your setup by running the JAX initialization test:

```bash
uv run pytest tests/test_jax_init.py
```
In the devcontainer, this will succeed on both CPU and GPU. A `GpuDevice` is expected if a GPU is detected and the `cuda` extra was installed.
