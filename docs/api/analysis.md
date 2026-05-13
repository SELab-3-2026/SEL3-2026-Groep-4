# Analysis & Plotting Tools

This guide outlines the tools available for analyzing experimental data and generating poster-quality visualizations for the Brittle Star project.

## Shared Configuration

All plotting scripts share a central configuration in `scripts/plots/plot_config.py`. This file defines:
- **Color Palette:** A color-blind friendly, high-contrast palette for different architectures.
- **Typography:** Consistent font sizes and styles tailored for A0 posters.
- **Markers:** Shared visual indicators, such as the ★ used for best performers.

## Comparison Visualization

The `scripts/plots/analyze_comparisons.py` script generates grouped bar charts comparing the performance of different architectures across various morphologies.

### Usage

Run the script from the root of the project, providing the path to your evaluation CSV:

```bash
# Basic usage (saves PNG and SVG to runs/evaluation/plots/)
uv run python scripts/plots/analyze_comparisons.py path/to/results.csv

# Advanced usage for Figma/Poster integration
uv run python scripts/plots/analyze_comparisons.py path/to/results.csv \
    --output_dir docs/assets/plots/ \
    --font_size 30 \
    --fig_width 14 \
    --fig_height 10
```

### CLI Arguments

- `input_csv`: (Required) Path to the CSV file containing evaluation results.
- `--output_dir`, `-o`: Directory where plots will be saved (default: `runs/evaluation/plots`).
- `--show_titles`: Include titles in the plots. Default is **False**, as titles are typically added natively in design tools like Figma.
- `--font_size`: Base font size in points (default: 28).
- `--fig_width` / `--fig_height`: Physical dimensions of the plot in inches. Match these to your Figma layout to maintain exact font sizes.

### Outputs

The script generates four key plots, each saved as both `.png` and `.svg`:
1.  **Forward Velocity:** Grouped bar chart (cm/s).
2.  **Accumulated Reward:** Mean cumulative reward.
3.  **Success Rate:** Target acquisition percentage.
4.  **Distance Remaining:** Navigational accuracy.

---

## Convergence Analysis

The `scripts/plots/analyze_convergence.py` script determines the convergence point of training runs.

### Usage

```bash
uv run python scripts/plots/analyze_convergence.py --output_dir runs/convergence/
```

### Configuration

- **File Mapping:** The script uses hardcoded paths in the `FILE_MAPPING` dictionary. Update these paths to point to your specific run evaluation files.
- **CLI Arguments:** Supports the same `--show_titles`, `--font_size`, and `--fig_width/height` flags as the comparison script.

### Outputs

Generates three plots (PNG & SVG):
1. `convergence_comparison`: Grouped horizontal bar chart.
2. `progress_reward_curves`: Line plots of reward over time.
3. `progress_velocity_curves`: Line plots of velocity over time.

---

## Poster Integration (Figma)

### SVG & Scaling
We recommend using the **SVG** outputs for poster design in Figma:
1. **No Resolution Loss:** SVGs are vector-based and will remain sharp at any size.
2. **Native Text:** Text in the SVG imports as native text layers in Figma.
3. **Exact Font Matching:** To ensure a `28pt` font in the plot matches a `28pt` font in your poster, set the `--fig_width` and `--fig_height` to match the physical dimensions of the plot box in your Figma layout.
4. **Editable:** You can "Ungroup" the SVG in Figma to manually move labels, adjust colors, or tweak individual bars.

### Image Placeholders
The comparison charts include light-gray square placeholders below the X-axis. These are designed as guides; in Figma, you can drop your morphology renders or illustrations directly on top of these squares.
