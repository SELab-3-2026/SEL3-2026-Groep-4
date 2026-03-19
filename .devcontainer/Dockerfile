FROM mcr.microsoft.com/devcontainers/python:3.12

# Install uv for dependency management
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

# Install system dependencies for JAX, OpenGL, and Mujoco
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libosmesa6-dev \
    libglew-dev \
    libglfw3 \
    patchelf \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# Set environment variables for hardware acceleration and rendering
ENV LD_LIBRARY_PATH=/usr/lib/nvidia
ENV NVIDIA_VISIBLE_DEVICES=all
ENV NVIDIA_DRIVER_CAPABILITIES=compute,utility,graphics

# Set the workspace path (defaults to /workspaces/project but can be overridden)
ARG WORKSPACE_PATH=/workspaces/project
ENV WORKSPACE_PATH=$WORKSPACE_PATH
ENV PATH="$WORKSPACE_PATH/.venv/bin:$PATH"

# Ensure the workspace and .venv directories exist and are owned by the vscode user
# This prevents permission errors when the postCreateCommand runs uv sync
RUN mkdir -p $WORKSPACE_PATH/.venv && chown -R vscode:vscode $WORKSPACE_PATH

USER vscode
WORKDIR $WORKSPACE_PATH
