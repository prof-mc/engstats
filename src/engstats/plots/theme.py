"""
engstats.plots.theme
=====================
Default visual theme applied automatically on `import engstats`.
Built on top of seaborn's theming system and matplotlib rcParams.
"""

import matplotlib.pyplot as plt
import seaborn as sns

# Colour palette — muted, accessible, works on white backgrounds
PALETTE = ["#2563EB", "#DC2626", "#16A34A", "#D97706", "#7C3AED", "#0891B2"]
BACKGROUND = "#FFFFFF"
GRID_COLOR = "#E5E7EB"
TEXT_COLOR = "#1F2937"


def apply_theme() -> None:
    """
    Apply the engstats default plot theme globally.

    Called automatically on ``import engstats``. Safe to call again
    if rcParams are reset by other libraries.
    """
    sns.set_theme(
        style="whitegrid",
        palette=PALETTE,
        font="DejaVu Sans",
        font_scale=1.1,
        rc={
            "axes.facecolor": BACKGROUND,
            "figure.facecolor": BACKGROUND,
            "axes.edgecolor": GRID_COLOR,
            "axes.labelcolor": TEXT_COLOR,
            "axes.titlesize": 13,
            "axes.labelsize": 11,
            "xtick.color": TEXT_COLOR,
            "ytick.color": TEXT_COLOR,
            "grid.color": GRID_COLOR,
            "grid.linewidth": 0.8,
            "lines.linewidth": 2.0,
            "patch.linewidth": 0.5,
            "figure.dpi": 110,
            "figure.figsize": (8, 5),
            "legend.framealpha": 0.9,
            "legend.fontsize": 10,
        },
    )
