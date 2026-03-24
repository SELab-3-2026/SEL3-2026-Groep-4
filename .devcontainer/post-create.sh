#!/bin/bash
set -e

# Detect if a GPU is available via nvidia-smi.
# This works for Linux hosts and Windows (WSL2) with NVIDIA Container Toolkit.
# On Mac (Apple Silicon) or systems without NVIDIA GPUs, this will skip the 'cuda' extra.
if command -v nvidia-smi &> /dev/null && nvidia-smi &> /dev/null; then
    echo "GPU detected. Syncing with 'cuda' extra..."
    uv sync --frozen --extra cuda
else
    echo "No GPU detected or nvidia-smi failed. Syncing without 'cuda' extra..."
    uv sync --frozen
fi

echo "Environment synced successfully."

# Install pre-commit hooks so they are active in the devcontainer
uv run pre-commit install
