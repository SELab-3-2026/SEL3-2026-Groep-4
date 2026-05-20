# Simulation & Evaluation

The simulation pipeline allows you to visualize trained models and evaluate their performance under various conditions.

## Overview

The simulation pipeline is metadata-driven. Training-specific configurations (morphology, arena, environment, etc.) are automatically loaded from the `_metadata.yaml` file associated with the model checkpoint.

## Basic Simulation

To simulate a model in the MuJoCo viewer:

```bash
uv run scripts/simulate.py simulation.model_path=runs/your_run/final_model.flax
```

## Amputation & Morphology Overrides

You can test trained models on different morphologies (e.g., amputating legs) by providing a morphology override. The observations will be automatically padded up to the training morphology's dimensions:

```bash
uv run scripts/simulate.py \
    simulation.model_path=runs/your_run/final_model.flax \
    simulation.morphology_override=configs/morphology/3_arms.yaml
```

## Video Recording

Recording videos requires the `[evaluation]` extra:

```bash
uv run scripts/simulate.py \
    simulation.model_path=runs/your_run/final_model.flax \
    simulation.record_video=true \
    simulation.max_steps=1000
```

Videos and evaluation metadata are stored in timestamped folders alongside the model:
`runs/your_run/final_model_evaluations/eval_<timestamp>/simulation.mp4`

### Top-Down and Follow Cameras

Using the following script, you can render a top-down and follow camera view for multiple models at once:

```bash
uv run scripts/poster_visualisations/render_poster_videos.py \
  runs/final-models/centralized/.../final_model.flax \
  runs/final-models/fully-connected/.../final_model.flax \
  runs/final-models/ring/.../final_model.flax \
  --max-steps 10000 --width 640 --height 480 --fps 60 \
  --output-root vids/poster/
```

For batch evaluation and cross-model comparison, see the **[Evaluation Guide](./evaluation.md)**.
