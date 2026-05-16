"""
Poster Comparison Visualizations

This script generates a Forward Velocity plot and three secondary plots (Accumulated Reward, Success
Rate, Distance Remaining).
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from plot_config import (
    COLORS,
    apply_style,
    BEST_PERFORMER_MARKER,
    BEST_PERFORMER_TEXT,
    BEST_PERFORMER_COLOR,
    create_common_parser,
    LEGEND_KWARGS,
)


def load_and_preprocess_data(filepath):
    """Loads CSV and prepares the metrics for plotting."""
    df = pd.read_csv(filepath)

    # Ensure success rate can be averaged numerically
    if "reached_target" in df.columns:
        df["reached_target"] = df["reached_target"].astype(int)

    return df


def _add_square_placeholders(ax, x_positions, labels):
    """Adds square placeholders for images below the x-axis."""
    for x, label in zip(x_positions, labels):
        # Create a roughly square rectangle in a mix of data/axes coords
        # Shifted down to avoid overlapping with x-tick labels
        rect = plt.Rectangle(
            (x - 0.25, -0.40),
            0.5,
            0.18,
            transform=ax.get_xaxis_transform(),
            facecolor="#F0F0F0",
            edgecolor="#A9A9A9",
            linestyle="--",
            zorder=1,
            clip_on=False,
        )
        ax.add_patch(rect)
        ax.text(
            x,
            -0.31,
            f"[ Insert {label}\nImage ]",
            transform=ax.get_xaxis_transform(),
            ha="center",
            va="center",
            fontsize=10,
            color="#888888",
            zorder=2,
        )


def plot_grouped_bar(
    df,
    metric_col,
    ylabel,
    title,
    output_filename,
    output_dir,
    higher_is_better=True,
    show_titles=False,
    figsize=(12, 8),
):
    """Generates and saves a highly customized grouped bar chart (grouped by Morphology)."""
    grouped = (
        df.groupby(["num_active_arms", "architecture"])[metric_col]
        .agg(["mean", "std"])
        .reset_index()
    )
    morphologies = sorted(grouped["num_active_arms"].unique(), reverse=True)
    architectures = grouped["architecture"].unique()

    fig, ax = plt.subplots(figsize=figsize)
    bar_width = 0.35
    x_indices = np.arange(len(morphologies))
    all_bars = {}
    all_means = []

    for i, arch in enumerate(architectures):
        arch_data = grouped[grouped["architecture"] == arch]
        means = [
            arch_data[arch_data["num_active_arms"] == m]["mean"].values[0]
            if not arch_data[arch_data["num_active_arms"] == m].empty
            else 0
            for m in morphologies
        ]
        stds = [
            arch_data[arch_data["num_active_arms"] == m]["std"].values[0]
            if not arch_data[arch_data["num_active_arms"] == m].empty
            else 0
            for m in morphologies
        ]
        all_means.extend(means)
        x_pos = x_indices + (i * bar_width) - (bar_width / 2 if len(architectures) == 2 else 0)
        color = COLORS.get(arch, "#888888")
        clean_label = arch.replace("_", " ").title()
        bars = ax.bar(
            x_pos,
            means,
            bar_width,
            yerr=stds,
            label=clean_label,
            color=color,
            capsize=8,
            error_kw={"elinewidth": 2, "alpha": 0.7},
        )
        all_bars[arch] = (x_pos, means, stds, bars)

    for m_idx, m in enumerate(morphologies):
        m_means = {arch: all_bars[arch][1][m_idx] for arch in architectures}
        best_arch = (
            max(m_means, key=m_means.get) if higher_is_better else min(m_means, key=m_means.get)
        )
        best_x = all_bars[best_arch][0][m_idx]
        best_y = all_bars[best_arch][1][m_idx]
        best_std = all_bars[best_arch][2][m_idx]
        offset = best_std + (abs(max(m_means.values())) * 0.05) if m_means.values() else 0
        ax.text(
            best_x,
            best_y + offset,
            BEST_PERFORMER_TEXT,
            ha="center",
            va="bottom",
            fontsize=28,
            color=BEST_PERFORMER_COLOR,
        )

    # Aesthetics
    ax.set_ylabel(ylabel, labelpad=15)
    if show_titles:
        ax.set_title(title, pad=25, fontweight="bold")

    x_ticks_pos = (
        x_indices
        + (bar_width / 2 if len(architectures) % 2 == 0 else 0)
        - (bar_width / 2 if len(architectures) == 2 else 0)
    )
    ax.set_xticks(x_ticks_pos)
    ax.set_xticklabels([f"{m} Arms" for m in morphologies])
    ax.tick_params(axis="x", pad=25)  # More padding for the squares

    # X-axis at zero
    ax.axhline(0, color="black", linewidth=1.5)
    ax.spines["bottom"].set_visible(False)

    # Y-axis limits explicitly including 0
    if all_means:
        min_val = min([*all_means, 0])
        max_val = max([*all_means, 0])
        margin = (max_val - min_val) * 0.15 if max_val != min_val else 0.1
        ax.set_ylim(min_val - margin, max_val + margin * 1.5)  # Extra top margin for stars
        # Format y-ticks to not have excessive decimals, include 0
        ticks = (
            [min_val, max_val]
            if min_val == 0 and max_val == 0
            else sorted(list(set([min_val, 0, max_val])))
        )
        ax.set_yticks(ticks)
        ax.yaxis.set_major_formatter(
            plt.FuncFormatter(lambda x, _: f"{x:.2f}" if abs(x) < 10 else f"{x:.0f}")
        )

    # _add_square_placeholders(ax, x_ticks_pos, [f"{m} Arms" for m in morphologies])

    # Add custom legend entry for best performer
    ax.plot(
        [],
        [],
        marker=BEST_PERFORMER_MARKER,
        color="w",
        markerfacecolor=BEST_PERFORMER_COLOR,
        markersize=15,
        label="Best Performance",
        ls="",
    )
    ax.legend(**LEGEND_KWARGS, ncol=len(architectures) + 1)
    ax.set_facecolor("white")
    fig.patch.set_facecolor("white")

    os.makedirs(output_dir, exist_ok=True)
    base_path = os.path.join(output_dir, os.path.splitext(output_filename)[0])
    plt.savefig(f"{base_path}.png", dpi=300, bbox_inches="tight")
    plt.savefig(f"{base_path}.svg", format="svg", bbox_inches="tight")
    plt.close()


def plot_grouped_bar_alt(
    df,
    metric_col,
    ylabel,
    title,
    output_filename,
    output_dir,
    higher_is_better=True,
    show_titles=False,
    figsize=(12, 8),
):
    """Generates and saves a highly customized grouped bar chart (grouped by Architecture)."""
    grouped = (
        df.groupby(["architecture", "num_active_arms"])[metric_col]
        .agg(["mean", "std"])
        .reset_index()
    )
    architectures = sorted(grouped["architecture"].unique())
    morphologies = sorted(grouped["num_active_arms"].unique(), reverse=True)

    fig, ax = plt.subplots(figsize=figsize)
    bar_width = 0.8 / len(morphologies)
    x_indices = np.arange(len(architectures))
    all_bars = {}
    all_means = []

    for i, m in enumerate(morphologies):
        m_data = grouped[grouped["num_active_arms"] == m]
        means = [
            m_data[m_data["architecture"] == arch]["mean"].values[0]
            if not m_data[m_data["architecture"] == arch].empty
            else 0
            for arch in architectures
        ]
        stds = [
            m_data[m_data["architecture"] == arch]["std"].values[0]
            if not m_data[m_data["architecture"] == arch].empty
            else 0
            for arch in architectures
        ]
        all_means.extend(means)

        # Offset bars based on morphology index
        offset = (i - len(morphologies) / 2 + 0.5) * bar_width
        x_pos = x_indices + offset

        # We can use a color gradient or different colors for morphologies
        # For simplicity, using a colormap
        color = plt.cm.viridis(i / max(1, len(morphologies) - 1))

        bars = ax.bar(
            x_pos,
            means,
            bar_width,
            yerr=stds,
            label=f"{m} Arms",
            color=color,
            capsize=4,
            error_kw={"elinewidth": 1.5, "alpha": 0.7},
        )
        all_bars[m] = (x_pos, means, stds, bars)

    for a_idx, arch in enumerate(architectures):
        a_means = {m: all_bars[m][1][a_idx] for m in morphologies}
        best_m = (
            max(a_means, key=a_means.get) if higher_is_better else min(a_means, key=a_means.get)
        )
        best_x = all_bars[best_m][0][a_idx]
        best_y = all_bars[best_m][1][a_idx]
        best_std = all_bars[best_m][2][a_idx]
        offset = best_std + (abs(max(a_means.values())) * 0.05) if a_means.values() else 0
        ax.text(
            best_x,
            best_y + offset,
            BEST_PERFORMER_TEXT,
            ha="center",
            va="bottom",
            fontsize=20,
            color=BEST_PERFORMER_COLOR,
        )

    # Aesthetics
    ax.set_ylabel(ylabel, labelpad=15)
    if show_titles:
        ax.set_title(title + " (Alt)", pad=25, fontweight="bold")

    ax.set_xticks(x_indices)
    ax.set_xticklabels([arch.replace("_", " ").title() for arch in architectures])
    ax.tick_params(axis="x", pad=25)

    # X-axis at zero
    ax.axhline(0, color="black", linewidth=1.5)
    ax.spines["bottom"].set_visible(False)

    if all_means:
        min_val = min([*all_means, 0])
        max_val = max([*all_means, 0])
        margin = (max_val - min_val) * 0.15 if max_val != min_val else 0.1
        ax.set_ylim(min_val - margin, max_val + margin * 1.5)
        ticks = (
            [min_val, max_val]
            if min_val == 0 and max_val == 0
            else sorted(list(set([min_val, 0, max_val])))
        )
        ax.set_yticks(ticks)
        ax.yaxis.set_major_formatter(
            plt.FuncFormatter(lambda x, _: f"{x:.2f}" if abs(x) < 10 else f"{x:.0f}")
        )

    # In this alt plot, placeholders might be per architecture
    _add_square_placeholders(
        ax, x_indices, [arch.replace("_", "\n").title() for arch in architectures]
    )

    ax.plot(
        [],
        [],
        marker=BEST_PERFORMER_MARKER,
        color="w",
        markerfacecolor=BEST_PERFORMER_COLOR,
        markersize=15,
        label="Best Performance",
        ls="",
    )
    ax.legend(**LEGEND_KWARGS, ncol=len(morphologies) + 1)
    ax.set_facecolor("white")
    fig.patch.set_facecolor("white")

    os.makedirs(output_dir, exist_ok=True)
    base_path = os.path.join(output_dir, os.path.splitext(output_filename)[0])
    plt.savefig(f"{base_path}.png", dpi=300, bbox_inches="tight")
    plt.savefig(f"{base_path}.svg", format="svg", bbox_inches="tight")
    plt.close()


if __name__ == "__main__":
    parser = create_common_parser(description="Generate comparison poster plots.")
    parser.add_argument(
        "input_csv", help="Path to the input CSV file containing evaluation results."
    )
    args = parser.parse_args()

    INPUT_CSV = args.input_csv
    OUTPUT_DIR = args.output_dir

    if not os.path.exists(INPUT_CSV):
        print(f"Error: Could not find {INPUT_CSV}. Please ensure the file exists.")
    else:
        df = load_and_preprocess_data(INPUT_CSV)
        print("Data loaded successfully. Generating poster plots...")

        apply_style(font_size=args.font_size)
        kwargs = {"show_titles": args.show_titles, "figsize": (args.fig_width, args.fig_height)}

        # Velocity Conversion: m/s to cm/s
        if "approx_max_velocity" in df.columns:
            df["approx_max_velocity"] = df["approx_max_velocity"] * 100

        # 1. Primary Plot: Forward Velocity
        plot_grouped_bar(
            df=df,
            metric_col="approx_max_velocity",
            ylabel="Max Forward Velocity (cm/s)",
            title="Graceful Degradation: Velocity Across Morphologies",
            output_filename="poster_plot_velocity.png",
            output_dir=OUTPUT_DIR,
            higher_is_better=True,
            **kwargs,
        )
        plot_grouped_bar_alt(
            df=df,
            metric_col="approx_max_velocity",
            ylabel="Max Forward Velocity (cm/s)",
            title="Graceful Degradation: Velocity Across Morphologies",
            output_filename="poster_plot_velocity_alt.png",
            output_dir=OUTPUT_DIR,
            higher_is_better=True,
            **kwargs,
        )

        # 2. Secondary Plot: Accumulated Reward
        plot_grouped_bar(
            df=df,
            metric_col="eval_return",
            ylabel="Mean Cumulative Reward",
            title="Overall Efficiency Across Morphologies",
            output_filename="poster_plot_reward.png",
            output_dir=OUTPUT_DIR,
            higher_is_better=True,
            **kwargs,
        )
        plot_grouped_bar_alt(
            df=df,
            metric_col="eval_return",
            ylabel="Mean Cumulative Reward",
            title="Overall Efficiency Across Morphologies",
            output_filename="poster_plot_reward_alt.png",
            output_dir=OUTPUT_DIR,
            higher_is_better=True,
            **kwargs,
        )

        # 3. Secondary Plot: Success Rate
        plot_grouped_bar(
            df=df,
            metric_col="reached_target",
            ylabel="Success Rate (%)",
            title="Target Acquisition Consistency",
            output_filename="poster_plot_success_rate.png",
            output_dir=OUTPUT_DIR,
            higher_is_better=True,
            **kwargs,
        )
        plot_grouped_bar_alt(
            df=df,
            metric_col="reached_target",
            ylabel="Success Rate (%)",
            title="Target Acquisition Consistency",
            output_filename="poster_plot_success_rate_alt.png",
            output_dir=OUTPUT_DIR,
            higher_is_better=True,
            **kwargs,
        )

        # 4. Secondary Plot: Final Distance Remaining
        plot_grouped_bar(
            df=df,
            metric_col="final_xy_dist",
            ylabel="Distance to Target Remaining",
            title="Navigational Accuracy (Lower is Better)",
            output_filename="poster_plot_distance.png",
            output_dir=OUTPUT_DIR,
            higher_is_better=False,  # For distance, a lower score is better
            **kwargs,
        )
        plot_grouped_bar_alt(
            df=df,
            metric_col="final_xy_dist",
            ylabel="Distance to Target Remaining",
            title="Navigational Accuracy (Lower is Better)",
            output_filename="poster_plot_distance_alt.png",
            output_dir=OUTPUT_DIR,
            higher_is_better=False,
            **kwargs,
        )

        print(f"All plots generated in the '{OUTPUT_DIR}/' directory.")
