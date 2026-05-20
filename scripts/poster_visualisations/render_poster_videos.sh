#!/usr/bin/env bash

# Multi-camera renders per model -> vids/poster/{arms}arms/{arch}/topdown.mp4 + follow.mp4

path=$1

uv run scripts/poster_visualisations/render_poster_videos.py \
  "$path"/centralized.flax \
  "$path"/fully-connected.flax \
  "$path"/ring.flax \