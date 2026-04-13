"""
engstats.plots.multivariate
=============================
Correlation heatmaps and pairplots.
"""

import matplotlib.pyplot as plt
import matplotlib.axes
import seaborn as sns
import pandas as pd


def plot_correlation_heatmap(
    data: pd.DataFrame,
    method: str = "pearson",
    annot: bool = True,
    title: str = "",
    ax: matplotlib.axes.Axes | None = None,
) -> matplotlib.axes.Axes:
    """
    Heatmap of the pairwise correlation matrix.

    Parameters
    ----------
    data : pd.DataFrame
    method : {'pearson', 'spearman', 'kendall'}
    annot : bool
        Show correlation values on the heatmap. Default True.
    title : str
    ax : matplotlib.axes.Axes, optional

    Returns
    -------
    matplotlib.axes.Axes
    """
    numeric = data.select_dtypes(include="number")
    if numeric.shape[1] < 2:
        raise ValueError("DataFrame must have at least 2 numeric columns.")
    corr = numeric.corr(method=method).round(2)
    if ax is None:
        figsize = (max(5, corr.shape[1]), max(4, corr.shape[0]))
        _, ax = plt.subplots(figsize=figsize)
    sns.heatmap(
        corr, annot=annot, fmt=".2f", cmap="coolwarm",
        vmin=-1, vmax=1, linewidths=0.5, ax=ax
    )
    ax.set_title(title or f"Correlation Matrix ({method.capitalize()})")
    plt.tight_layout()
    return ax


def plot_pairplot(
    data: pd.DataFrame,
    hue: str | None = None,
    diag_kind: str = "hist",
    title: str = "",
) -> sns.PairGrid:
    """
    Seaborn pairplot for all numeric columns.

    Parameters
    ----------
    data : pd.DataFrame
    hue : str, optional
        Categorical column to colour points by.
    diag_kind : {'kde', 'hist'}
        Type of plot on the diagonal.
    title : str

    Returns
    -------
    seaborn.PairGrid
    """
    g = sns.pairplot(data, hue=hue, diag_kind=diag_kind, plot_kws={"alpha": 0.6})
    if title:
        g.figure.suptitle(title, y=1.02)
    plt.tight_layout()
    return g
