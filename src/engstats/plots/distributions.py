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
    kde: bool = True,
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
