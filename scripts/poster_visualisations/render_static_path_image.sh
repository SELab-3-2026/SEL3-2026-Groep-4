#!/usr/bin/env bash

# Static path image + optional ghost render

path=$1 # path to .flax model with metadata.yaml alongside it

uv run scripts/poster_visualisations/render_static_path_image.py \
  "$path" \
  --output-path vids/poster/5arms/centralized/path.png \
  --ghost-overlay
