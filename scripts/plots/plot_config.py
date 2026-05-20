import argparse
import matplotlib.pyplot as plt

# Shared Color Palette (Colorblind friendly, high contrast)
# Matches poster design
COLORS = {
    "CENTRALIZED": "#0D567C",  # Blue
    "FULLY_CONNECTED": "#8C0E0F",  # Reddish
    "RING": "#FCB305",  # Pale Yellow
}


def apply_style(font_size=36):
    """
    Applies the shared typography and aesthetic settings to Matplotlib.
    """
    plt.rcParams.update(
        {
            "font.size": font_size,
            "axes.labelsize": font_size,
            "axes.titlesize": font_size,
            "xtick.labelsize": font_size,
            "ytick.labelsize": font_size,
            "legend.fontsize": font_size,
            "axes.linewidth": 2,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.spines.left": False,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.bbox": "tight",
            "savefig.dpi": 300,
        }
    )


# Star marker for best performer
BEST_PERFORMER_TEXT = "★"
BEST_PERFORMER_MARKER = "*"
BEST_PERFORMER_COLOR = "#D4AF37"  # Gold

# Centralized Legend Configuration
LEGEND_KWARGS = {
    "loc": "upper center",
    "bbox_to_anchor": (0.5, -0.12),
    "frameon": False,
}


def create_common_parser(description: str) -> argparse.ArgumentParser:
    """
    Creates an argparse parser with common plotting arguments.
    """
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument(
        "--output_dir",
        "-o",
        default="runs/evaluation/plots",
        help="Directory to save the generated plots.",
    )
    parser.add_argument(
        "--show_titles",
        action="store_true",
        help="Include titles in the plots. Default is False for easier poster integration.",
    )
    parser.add_argument(
        "--font_size", type=int, default=28, help="Base font size in points. Default is 28."
    )
    parser.add_argument(
        "--fig_width", type=float, default=12.0, help="Figure width in inches. Default is 12.0."
    )
    parser.add_argument(
        "--fig_height", type=float, default=8.0, help="Figure height in inches. Default is 8.0."
    )
    return parser
