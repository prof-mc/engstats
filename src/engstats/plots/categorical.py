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
    Bar chart of the mean of ``y`` for each category of ``x``.

    Each bar's height is the group **mean** (not a count or a sum), and the
    black error bar is its 95 % confidence interval (seaborn default). For
    counts, use [`plot_pareto`][engstats.plots.distributions.plot_pareto].

    Parameters
    ----------
    data : pd.DataFrame
        Source DataFrame.
    x : str
        Categorical column (one bar per category).
    y : str
        Numeric column to average.
    title : str
        Plot title. Defaults to ``"<y> by <x>"``.
    xlabel, ylabel : str
        Axis labels. Default to the column names.
    color : str, optional
        Single colour for every bar.
    ax : matplotlib.axes.Axes, optional
        Existing axes to draw on.

    Returns
    -------
    matplotlib.axes.Axes

    Examples
    --------
    >>> df = es.load_dataset("bridges")
    >>> es.plot_bar(df, x="material", y="load_capacity_kn")
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
    Grouped (side-by-side) bar chart of the mean of ``y``.

    As in [`plot_bar`][engstats.plots.categorical.plot_bar], each bar is a group mean with a 95 % confidence
    interval error bar.

    Parameters
    ----------
    data : pd.DataFrame
        Source DataFrame.
    x : str
        Primary categorical variable (positions on the x-axis).
    y : str
        Numeric column to average.
    hue : str
        Secondary categorical variable; one coloured bar per level at each
        ``x`` position.
    title : str
        Plot title. Defaults to ``"<y> by <x> and <hue>"``.
    ax : matplotlib.axes.Axes, optional
        Existing axes to draw on.

    Returns
    -------
    matplotlib.axes.Axes

    Examples
    --------
    >>> df = es.load_dataset("circuits")
    >>> es.plot_grouped_bar(df, x="nominal_ohm", y="measured_ohm", hue="batch")
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

def plot_stackeddot(
        data: pd.DataFrame,
        x: str,
        ax: matplotlib.axes.Axes | None = None,
):
    """
    Stacked dot plot: one dot per observation, stacked over repeated values.

    Best for small samples of discrete or rounded values, where each stack
    height is the count at that value.

    Parameters
    ----------
    data : pd.DataFrame
        Source DataFrame. **Modified in place:** a ``stack`` column holding
        each dot's height is added. Pass ``df.copy()`` to keep ``df``
        unchanged.
    x : str
        Column to plot.
    ax : matplotlib.axes.Axes, optional
        Existing axes to draw on. A new 8 × 3 figure is created otherwise.

    Returns
    -------
    matplotlib.axes.Axes

    Examples
    --------
    >>> df = es.load_dataset("concrete")
    >>> es.plot_stackeddot(df.copy(), x="age_days")
    """
    data["stack"] = (data.groupby(x).cumcount() + 1) / 20

    if ax is None:
        _, ax = plt.subplots(figsize=(8, 3))
    sns.scatterplot(data=data, x=x, y="stack", s=70, ax=ax)

    ymax = data["stack"].max() * 2

    ax.set_ylim(0, ymax)

    ax.set_title(f"Stacked Dotplot of {x}")
    ax.set_xlabel(x)
    ax.set_ylabel("")
    ax.set_yticks([])
    plt.tight_layout()
    return ax

