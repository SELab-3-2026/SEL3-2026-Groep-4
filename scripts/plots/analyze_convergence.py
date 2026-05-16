"""
Convergence Analysis Script for Poster Visualizations

This script analyzes evaluation metrics from multiple training runs to determine
the convergence point of different reinforcement learning architectures.

Workflow:
1. Loads evaluation data from the CSV files defined in FILE_MAPPING.
2. Calculates a rolling average of the reward and velocity to smooth noise.
3. Determines the convergence timestep for each metric (first time 95% of peak is reached).
4. Generates a grouped bar chart comparing convergence speed and line plots of the raw curves.

Usage:
    uv run python scripts/analysis/analyze_convergence.py

Note: For these metrics to be valid, the evaluation CSVs must be generated with
exploration noise strictly disabled (e.g., taking the mean of the action distribution).
"""

import logging
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from enum import Enum

from plot_config import COLORS, apply_style, create_common_parser, LEGEND_KWARGS

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# --- Globals & Configuration ---
USING_DUMMY_DATA = False
SMOOTHING_WINDOW = 3
CONVERGENCE_THRESHOLD = 0.95


class Columns(str, Enum):
    # ... (rest of the file remains same, just need to update plotting functions and obtain_data)
    """Column names expected in every evaluation CSV."""

    ARCH = "architecture"
    TIMESTEPS = "trained_timesteps"
    REWARD = "eval_return"
    VELOCITY = "velocity"
    EVAL_STEPS = "eval_steps"
    FINAL_XY_DIST = "final_xy_dist"
    INITIAL_XY_DIST = "initial_xy_dist"
    REACHED_TARGET = "reached_target"


# Maps architecture display names to the path of their evaluation CSV.
# Update these paths once real evaluation data is available.
FILE_MAPPING: dict[str, str] = {
    "centralized 2 arms": "runs/dummy/dummy_centralized_2_arms.csv",
    "centralized 5 arms": "runs/dummy/dummy_centralized_5_arms.csv",
    "decentralized fully connected": "runs/dummy/dummy_decentralized_fully_connected.csv",
    "decentralized ring-level": "runs/dummy/dummy_decentralized_ring-level.csv",
    "decentralized segment-level": "runs/dummy/dummy_decentralized_segment-level.csv",
}

# Architecture profiles for dummy data generation: (max_reward, max_velocity, sigmoid_speed)
_DUMMY_PROFILES: dict[str, tuple[float, float, float]] = {
    "centralized 2 arms": (300, 0.8, 1.2),
    "centralized 5 arms": (450, 1.1, 1.0),
    "decentralized fully connected": (500, 1.3, 0.7),
    "decentralized ring-level": (480, 1.2, 0.8),
    "decentralized segment-level": (520, 1.4, 0.6),
}


def generate_dummy_csvs(file_mapping: dict[str, str]):
    """
    Generates one dummy CSV per architecture in FILE_MAPPING at their expected locations.
    Skips any architecture without a defined profile.
    """
    checkpoints = list(range(100, 1100, 100))
    timesteps = [cp * 10_000 for cp in checkpoints]

    for arch, path in file_mapping.items():
        if arch not in _DUMMY_PROFILES:
            logger.warning(f"No dummy profile for '{arch}'. Skipping.")
            continue

        m_reward, m_vel, speed = _DUMMY_PROFILES[arch]

        rows = []
        for i, ts in enumerate(timesteps):
            progress = 1 / (1 + np.exp(-speed * (i - 4)))
            rows.append(
                {
                    Columns.TIMESTEPS: ts,
                    Columns.REWARD: m_reward * progress + np.random.normal(0, 5),
                    Columns.VELOCITY: m_vel * progress + np.random.normal(0, 0.02),
                }
            )

        # Create parent directories if they don't exist
        os.makedirs(os.path.dirname(path), exist_ok=True)

        pd.DataFrame(rows).to_csv(path, index=False)
        logger.info(f"Generated dummy CSV at expected path: {path}")


def load_metrics(file_mapping: dict[str, str]) -> pd.DataFrame:
    """
    Loads one CSV per architecture, injects the architecture name as a column,
    and returns the combined DataFrame with only the required columns.
    """
    required = [
        Columns.CHECKPOINT,
        Columns.TIMESTEPS,
        Columns.REWARD,
        Columns.INITIAL_XY_DIST,
        Columns.FINAL_XY_DIST,
        Columns.EVAL_STEPS,
    ]
    dfs = []

    for arch_name, filepath in file_mapping.items():
        if not os.path.exists(filepath):
            logger.warning(f"File not found: '{filepath}'. Skipping.")
            continue

        df = pd.read_csv(filepath)

        missing = [c for c in required if c not in df.columns]
        if missing:
            logger.warning(f"Missing columns {missing} in '{filepath}'. Skipping.")
            continue

        df = df[required].copy()
        df[Columns.ARCH] = arch_name
        dfs.append(df)

    return pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame()


def _convergence_timestep(
    series: pd.Series, timesteps: pd.Series, checkpoints: pd.Series
) -> tuple[float, int, int]:
    """Returns the first timestep where the smoothed series reaches 95% of its peak."""
    smoothed = series.rolling(window=SMOOTHING_WINDOW, min_periods=1).mean()
    threshold = smoothed.max() * CONVERGENCE_THRESHOLD

    mask = smoothed >= threshold
    first_idx = mask.idxmax()

    return timesteps.loc[first_idx], first_idx, checkpoints.loc[first_idx]


def analyze_convergence(df: pd.DataFrame) -> pd.DataFrame:
    """
    For each architecture, determines the convergence timestep based on both
    reward and velocity, returning one summary row per architecture.
    """
    results = []

    for arch in df[Columns.ARCH].unique():
        arch_data = df[df[Columns.ARCH] == arch].sort_values(Columns.TIMESTEPS)

        reward_timestep, reward_checkpoint_idx, reward_checkpoint = _convergence_timestep(
            arch_data[Columns.REWARD],
            arch_data[Columns.TIMESTEPS],
            arch_data[Columns.CHECKPOINT],
        )

        velocity_timestep, velocity_checkpoint_idx, velocity_checkpoint = _convergence_timestep(
            arch_data[Columns.VELOCITY],
            arch_data[Columns.TIMESTEPS],
            arch_data[Columns.CHECKPOINT],
        )

        results.append(
            {
                "Architecture": arch,
                "Reward_Convergence_Timestep": reward_timestep,
                "Reward_Convergence_Checkpoint_Idx": reward_checkpoint_idx,
                "Reward_Convergence_Checkpoint": reward_checkpoint,
                "Velocity_Convergence_Timestep": velocity_timestep,
                "Velocity_Convergence_Checkpoint_Idx": velocity_checkpoint_idx,
                "Velocity_Convergence_Checkpoint": velocity_checkpoint,
            }
        )

    return pd.DataFrame(results)


def _add_bar_labels(bars, max_val: float):
    """Annotates each bar with its value in white bold text, positioned inside."""
    for bar in bars:
        width = bar.get_width()
        label = f"{width / 1e6:.1f}M" if width >= 1e6 else f"{width:,.0f}"
        plt.text(
            width - (max_val * 0.02),
            bar.get_y() + bar.get_height() / 2,
            label,
            ha="right",
            va="center",
            fontsize=11,
            color="white",
            fontweight="bold",
        )


def plot_grouped_convergence_chart(
    results_df: pd.DataFrame, output_filename: str, output_dir: str, **kwargs
):
    """
    Saves a grouped horizontal bar chart comparing Reward and Velocity convergence timesteps
    across all architectures.
    """
    sorted_df = results_df.sort_values("Reward_Convergence_Timestep", ascending=True)
    architectures = sorted_df["Architecture"].tolist()
    y_pos = np.arange(len(architectures))
    bar_height = 0.35
    max_val = sorted_df[
        ["Reward_Convergence_Timestep", "Velocity_Convergence_Timestep"]
    ].values.max()

    fig, ax = plt.subplots(figsize=kwargs.get("figsize", (12, 8)))

    bars_reward = ax.barh(
        y_pos + bar_height / 2,
        sorted_df["Reward_Convergence_Timestep"],
        height=bar_height,
        label="Reward Convergence",
        color="#1f77b4",
    )
    bars_velocity = ax.barh(
        y_pos - bar_height / 2,
        sorted_df["Velocity_Convergence_Timestep"],
        height=bar_height,
        label="Velocity Convergence",
        color="#ff7f0e",
    )

    title_suffix = " (DUMMY DATA)" if USING_DUMMY_DATA else ""
    if kwargs.get("show_titles", True):
        ax.set_title(
            f"Comparison of Training Convergence Timesteps{title_suffix}", fontsize=20, pad=20
        )
    ax.set_xlabel("Timesteps to Convergence (95% of peak)", fontsize=16)
    ax.set_ylabel("Architecture", fontsize=16)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(architectures, fontsize=14)
    ax.tick_params(axis="x", labelsize=14)
    ax.legend(**LEGEND_KWARGS, ncol=2)
    ax.set_xlim(left=0)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    _add_bar_labels(bars_reward, max_val)
    _add_bar_labels(bars_velocity, max_val)

    plt.tight_layout()
    os.makedirs(output_dir, exist_ok=True)
    base_path = os.path.join(output_dir, os.path.splitext(output_filename)[0])
    plt.savefig(f"{base_path}.png", dpi=300, bbox_inches="tight")
    plt.savefig(f"{base_path}.svg", format="svg", bbox_inches="tight")
    plt.close()


def plot_metric_curves(
    df: pd.DataFrame, metric_col: str, title: str, output_filename: str, output_dir: str, **kwargs
):
    """
    Saves a line plot of the given metric over training timesteps for every architecture.
    """
    fig, ax = plt.subplots(figsize=kwargs.get("figsize", (12, 7)))

    for arch in df[Columns.ARCH].unique():
        arch_data = df[df[Columns.ARCH] == arch].sort_values(Columns.TIMESTEPS)
        color_key = arch.split()[0].upper() if isinstance(arch, str) else "UNKNOWN"
        color = COLORS.get(color_key, "#888888")
        ax.plot(
            arch_data[Columns.TIMESTEPS],
            arch_data[metric_col],
            label=arch,
            marker="o",
            markersize=4,
            alpha=0.8,
            color=color,
        )

    title_suffix = " (DUMMY DATA)" if USING_DUMMY_DATA else ""
    if kwargs.get("show_titles", True):
        ax.set_title(f"{title}{title_suffix}", fontsize=18, pad=20)
    ax.set_xlabel("Training Timesteps", fontsize=14)
    ax.set_ylabel(metric_col.replace("_", " ").title(), fontsize=14)
    ax.legend(**LEGEND_KWARGS, ncol=len(df[Columns.ARCH].unique()))
    ax.grid(True, linestyle="--", alpha=0.6)
    ax.set_xlim(left=0)
    ax.set_ylim(bottom=0)

    plt.tight_layout()
    os.makedirs(output_dir, exist_ok=True)
    base_path = os.path.join(output_dir, os.path.splitext(output_filename)[0])
    plt.savefig(f"{base_path}.png", dpi=300, bbox_inches="tight")
    plt.savefig(f"{base_path}.svg", format="svg", bbox_inches="tight")
    plt.close()


def plot_results(df: pd.DataFrame, results: pd.DataFrame, output_dir: str, **kwargs):
    """Generates and saves all analysis plots."""
    plot_grouped_convergence_chart(
        results, output_filename="convergence_comparison.png", output_dir=output_dir, **kwargs
    )
    plot_metric_curves(
        df,
        Columns.REWARD,
        "Training Progress: Accumulated Reward",
        "progress_reward_curves.png",
        output_dir=output_dir,
        **kwargs,
    )
    plot_metric_curves(
        df,
        Columns.VELOCITY,
        "Training Progress: Velocity",
        "progress_velocity_curves.png",
        output_dir=output_dir,
        **kwargs,
    )


def obtain_data() -> pd.DataFrame:
    """Resolves the file mapping, falling back to generated dummy CSVs if needed."""
    global USING_DUMMY_DATA
    if not any(os.path.exists(p) for p in FILE_MAPPING.values()):
        logger.info("No real evaluation files found. Generating dummy CSVs at expected locations.")
        generate_dummy_csvs(FILE_MAPPING)
        USING_DUMMY_DATA = True

    return load_metrics(FILE_MAPPING)


def run_analysis(output_dir: str, **kwargs):
    """Orchestrates data loading, convergence analysis, and plot generation."""
    df = obtain_data()
    if df.empty:
        logger.error("No data found to analyze.")
        return

    results = analyze_convergence(df)
    print(
        results[
            ["Architecture", "Reward_Convergence_Checkpoint_Idx", "Reward_Convergence_Checkpoint"]
        ]
    )

    plot_results(df, results, output_dir, **kwargs)
    logger.info("Analysis complete. Plots saved to disk.")


if __name__ == "__main__":
    parser = create_common_parser(description="Analyze training convergence.")
    args = parser.parse_args()

    apply_style(font_size=args.font_size)
    run_analysis(
        output_dir=args.output_dir,
        show_titles=args.show_titles,
        figsize=(args.fig_width, args.fig_height),
    )
