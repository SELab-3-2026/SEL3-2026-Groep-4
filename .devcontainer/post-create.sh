#!/bin/bash
set -e

# Determine if we should use CUDA or ROCm extras based on the devcontainer type or hardware detection.
IF_CUDA=false
IF_ROCM=false

if [ "$DEVCONTAINER_TYPE" = "cuda" ]; then
    echo "CUDA devcontainer detected. Forcing 'cuda' extra..."
    IF_CUDA=true
elif [ "$DEVCONTAINER_TYPE" = "rocm" ]; then
    echo "ROCm devcontainer detected. Forcing 'rocm' extra..."
    IF_ROCM=true
elif [ "$DEVCONTAINER_TYPE" = "cpu" ]; then
    echo "CPU devcontainer detected. Skipping hardware extras..."
else
    # Automatic detection fallback
    if command -v nvidia-smi &> /dev/null && nvidia-smi &> /dev/null; then
        echo "NVIDIA GPU detected via nvidia-smi. Syncing with 'cuda' extra..."
        IF_CUDA=true
    elif command -v rocminfo &> /dev/null || command -v rocm-smi &> /dev/null; then
        echo "AMD GPU detected via ROCm tools. Syncing with 'rocm' extra..."
        IF_ROCM=true
    fi
fi

if [ "$IF_CUDA" = "true" ]; then
    uv sync --frozen --extra cuda
elif [ "$IF_ROCM" = "true" ]; then
    uv sync --frozen --extra rocm
else
    echo "Syncing without hardware extras..."
    uv sync --frozen
fi

echo "Environment synced successfully."

# Install pre-commit hooks so they are active in the devcontainer
uv run pre-commit install
