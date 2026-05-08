#!/usr/bin/env bash

path=$1

uv run scripts/simulate.py \
    simulation.model_path="$path"/final_model.flax \
    simulation.record_video=True \
    simulation.video_output_path=./vids/simulation.mp4 \
    simulation.max_steps=10000