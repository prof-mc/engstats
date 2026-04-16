"""
engstats.plots.distributions
==============================
Univariate distribution plots built on seaborn + matplotlib.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.axes
import seaborn as sns
import pandas as pd
from engstats.utils.validation import require_numeric_1d


def plot_histogram(
    data,
    bins: int | str = "auto",
    title: str = "",
    xlabel: str = "Value",
    kde: bool = False,
    color: str | None = None,
    ax: matplotlib.axes.Axes | None = None,
) -> matplotlib.axes.Axes:
    """
    Plot a histogram with an optional KDE overlay.

    Parameters
    ----------
    data : array-like
    bins : int or 'auto'
    title : str
    xlabel : str
    kde : bool
        Overlay a kernel density estimate. Default True.
    color : str, optional
    ax : matplotlib.axes.Axes, optional
        Existing axes to draw on.

    Returns
    -------
    matplotlib.axes.Axes
    """
    arr = require_numeric_1d(data, "data")
    if ax is None:
        _, ax = plt.subplots()
    sns.histplot(arr, bins=bins, kde=kde, color=color, ax=ax)
    ax.set_title(title or "Histogram")
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Count")
    plt.tight_layout()
    return ax


def plot_kde(
    data,
    title: str = "",
    xlabel: str = "Value",
    fill: bool = True,
    color: str | None = None,
    ax: matplotlib.axes.Axes | None = None,
) -> matplotlib.axes.Axes:
    """
    Plot a kernel density estimate.

    Parameters
    ----------
    data : array-like
    title, xlabel : str
    fill : bool
        Shade area under the KDE curve.
    color : str, optional
    ax : matplotlib.axes.Axes, optional
    """
    arr = require_numeric_1d(data, "data")
    if ax is None:
        _, ax = plt.subplots()
    sns.kdeplot(arr, fill=fill, color=color, ax=ax)
    ax.set_title(title or "Density Estimate")
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Density")
    plt.tight_layout()
    return ax


def plot_boxplot(
    data: pd.DataFrame | None = None,
    x: str | None = None,
    y: str | None = None,
    values=None,
    title: str = "",
    ax: matplotlib.axes.Axes | None = None,
) -> matplotlib.axes.Axes:
    """
    Plot a box-and-whisker plot.

    Usage A — single array::

        es.plot_boxplot(values=my_array, title="Load measurements")

    Usage B — DataFrame with grouping::

        es.plot_boxplot(df, x="material", y="strength")
    """
    if ax is None:
        _, ax = plt.subplots()
    if values is not None:
        arr = require_numeric_1d(values, "values")
        sns.boxplot(y=arr, ax=ax)
        ax.set_xlabel("")
    elif data is not None and y is not None:
        sns.boxplot(data=data, x=x, y=y, ax=ax)
    else:
        raise ValueError("Provide either 'values' or both 'data' and 'y'.")
    ax.set_title(title or "Box Plot")
    plt.tight_layout()
    return ax


def plot_violin(
    data: pd.DataFrame | None = None,
    x: str | None = None,
    y: str | None = None,
    values=None,
    title: str = "",
    ax: matplotlib.axes.Axes | None = None,
) -> matplotlib.axes.Axes:
    """
    Plot a violin plot (combines box plot with KDE).

    Usage mirrors :func:`plot_boxplot`.
    """
    if ax is None:
        _, ax = plt.subplots()
    if values is not None:
        arr = require_numeric_1d(values, "values")
        sns.violinplot(y=arr, ax=ax)
    elif data is not None and y is not None:
        sns.violinplot(data=data, x=x, y=y, ax=ax)
    else:
        raise ValueError("Provide either 'values' or both 'data' and 'y'.")
    ax.set_title(title or "Violin Plot")
    plt.tight_layout()
    return ax


def plot_ecdf(
    data,
    title: str = "",
    xlabel: str = "Value",
    color: str | None = None,
    ax: matplotlib.axes.Axes | None = None,
) -> matplotlib.axes.Axes:
    """
    Plot the empirical cumulative distribution function (ECDF).

    Parameters
    ----------
    data : array-like
    title, xlabel : str
    """
    arr = require_numeric_1d(data, "data")
    if ax is None:
        _, ax = plt.subplots()
    sns.ecdfplot(arr, color=color, ax=ax)
    ax.set_title(title or "ECDF")
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Proportion")
    plt.tight_layout()
    return ax


def plot_pareto(
    data: pd.DataFrame,
    x: str,
    title: str = "",
    xlabel: str = "",
    ax: matplotlib.axes.Axes | None = None,
) -> matplotlib.axes.Axes:
    """
    Plot a Pareto chart for a categorical column.

    Parameters
    ----------
    data : pd.DataFrame
        Source DataFrame.
    x : str
        Categorical column to summarize.
    threshold : float
        Reference cumulative proportion, default 0.80.
    title : str
        Plot title.
    xlabel : str
        X-axis label.
    ax : matplotlib.axes.Axes, optional
        Existing axes to draw on.

    Returns
    -------
    matplotlib.axes.Axes
    """
    if x not in data.columns:
        raise ValueError(f"Column '{x}' not found in DataFrame.")

    values = data[x].dropna()
    if values.empty:
        raise ValueError(f"Column '{x}' has no non-null values.")

    counts = values.value_counts().sort_values(ascending=False)
    cumperc = counts.cumsum() / counts.sum() * 100

    if ax is None:
        _, ax = plt.subplots(figsize=(8, 4))

    ax2 = ax.twinx()

    sns.barplot(x=counts.index, y=counts.values, ax=ax)
    ax2.plot(range(len(counts)), cumperc.values, marker="o", linestyle="--", color="red")
    ax2.grid(linestyle=":")

    ax.set_title(title or f"Pareto Plot: {x}")
    ax.set_xlabel(xlabel or x)
    ax.set_ylabel("Count")

    ax2.set_ylabel("Cumulative %")
    ax2.set_ylim(0, 105)

    return ax


def plot_side_by_side(
    data: pd.DataFrame,
    x: list[str],
    style: ["individual", "box"] = "individual",
    ax: matplotlib.axes.Axes | None = None,
):
    if ax is None:
        _, ax = plt.subplots(figsize=(8, 4))

    if isinstance(x, str):
        x = [x]
    if style == "individual":
        plot_mech = sns.swarmplot
    elif style == "box":
        plot_mech = sns.boxplot
    else:
        raise ValueError(f"Style '{style}' not supported.")

    subdf = data.loc[:, data.columns.intersection(x)]

    plot_mech(
        data=subdf.melt(var_name="Variable", value_name="Value"),
        x="Variable",
        y="Value",
        ax=ax,
    )

    ax.set_title(f'Side-by-side {style} plots')

    return ax
