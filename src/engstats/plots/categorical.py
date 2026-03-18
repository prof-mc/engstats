"""
engstats.plots.categorical
============================
Bar charts and grouped bar charts.
"""

import matplotlib.pyplot as plt
import matplotlib.axes
import seaborn as sns
import pandas as pd


def plot_bar(
    data: pd.DataFrame,
    x: str,
    y: str,
    title: str = "",
    xlabel: str = "",
    ylabel: str = "",
    color: str | None = None,
    ax: matplotlib.axes.Axes | None = None,
) -> matplotlib.axes.Axes:
    """
    Simple bar chart.

    Parameters
    ----------
    data : pd.DataFrame
    x : str
        Categorical column (groups).
    y : str
        Numeric column (heights).
    """
    if ax is None:
        _, ax = plt.subplots()
    sns.barplot(data=data, x=x, y=y, color=color, ax=ax)
    ax.set_title(title or f"{y} by {x}")
    ax.set_xlabel(xlabel or x)
    ax.set_ylabel(ylabel or y)
    plt.tight_layout()
    return ax


def plot_grouped_bar(
    data: pd.DataFrame,
    x: str,
    y: str,
    hue: str,
    title: str = "",
    ax: matplotlib.axes.Axes | None = None,
) -> matplotlib.axes.Axes:
    """
    Grouped (side-by-side) bar chart.

    Parameters
    ----------
    data : pd.DataFrame
    x : str
        Primary categorical variable.
    y : str
        Numeric response.
    hue : str
        Secondary grouping variable (produces side-by-side bars).
    """
    if ax is None:
        _, ax = plt.subplots()
    sns.barplot(data=data, x=x, y=y, hue=hue, ax=ax)
    ax.set_title(title or f"{y} by {x} and {hue}")
    ax.set_xlabel(x)
    ax.set_ylabel(y)
    ax.legend(title=hue)
    plt.tight_layout()
    return ax
