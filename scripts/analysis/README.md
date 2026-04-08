# Experiment Analysis Tools

This directory contains scripts for post-processing and analyzing experiment results, including TensorBoard logs and saved model weights.

## Scripts

### 1. `explore_tensorboard.py`
A CLI tool to summarize TensorBoard `tfevents` files without a GUI.

**Key Features:**
- Displays last values, min, max, and step counts for all scalar metrics.
- Calculates total run duration and estimated completion percentage.
- Exports granular scalar data to CSV for analysis in Excel/Pandas.

**Usage:**
```bash
# General usage
python explore_tensorboard.py <run_directory>

# Exporting data
python explore_tensorboard.py <run_directory> --csv data.csv
```

**Requirements:**
- `pandas`
- `tensorboard`
- `tensorflow-cpu` (or `tensorflow`)
