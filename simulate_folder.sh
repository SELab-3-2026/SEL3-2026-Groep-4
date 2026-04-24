#!/usr/bin/env bash
set -euo pipefail

# Usage: ./simulate_folder.sh /path/to/runs

BASE_DIR="${1:?Usage: $0 /path/to/runs}"

for run_dir in "$BASE_DIR"/*/; do
    echo "======================================"
    echo "Processing: $run_dir"

    # Find model and config files (first match in folder)
    flax_file=$(find "$run_dir" -maxdepth 1 -type f -name "*.flax" | head -n 1)
    yaml_file=$(find "$run_dir" -maxdepth 1 -type f -name "*.yaml" | head -n 1)

    # Safety checks
    if [[ -z "${flax_file:-}" ]]; then
        echo "❌ No .flax file found in $run_dir, skipping."
        continue
    fi

    if [[ -z "${yaml_file:-}" ]]; then
        echo "❌ No .yaml file found in $run_dir, skipping."
        continue
    fi

    echo "Using model:  $flax_file"
    echo "Using config: $yaml_file"

    output_file="$run_dir/simulation_output.log"

    echo "Running simulation..."

    uv run python scripts/simulate.py \
        simulation.model_path="$flax_file" \
        simulation.trained_config_path="$yaml_file" \
        simulation.headless=True \
        simulation.max_steps=5000 \
        > "$output_file" 2>&1

    echo "Saved output to: $output_file"
done