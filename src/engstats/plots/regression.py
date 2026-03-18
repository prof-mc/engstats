"""
engstats.plots.regression
===========================
Diagnostic and fitted-line plots for regression models.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.axes
import seaborn as sns
import pandas as pd
from scipy import stats


def plot_scatter_regression(
    data: pd.DataFrame,
    x: str,
    y: str,
    degree: int = 1,
    ci: int = 95,
    title: str = "",
    ax: matplotlib.axes.Axes | None = None,
) -> matplotlib.axes.Axes:
    """
    Scatter plot with a fitted regression line (and confidence band).

    Parameters
    ----------
    data : pd.DataFrame
    x, y : str
        Column names.
    degree : int
        Polynomial degree. 1 = straight line. Default 1.
    ci : int
        Confidence interval level (0–100). Default 95.
    title : str
    ax : matplotlib.axes.Axes, optional

    Returns
    -------
    matplotlib.axes.Axes
    """
    if ax is None:
        _, ax = plt.subplots()
    if degree == 1:
        sns.regplot(data=data, x=x, y=y, ci=ci, ax=ax,
                    scatter_kws={"alpha": 0.6}, line_kws={"linewidth": 2})
    else:
        sns.regplot(data=data, x=x, y=y, ci=ci, order=degree, ax=ax,
                    scatter_kws={"alpha": 0.6}, line_kws={"linewidth": 2})
    ax.set_title(title or f"Scatter Plot: {y} vs {x}")
    ax.set_xlabel(x)
    ax.set_ylabel(y)
    plt.tight_layout()
    return ax


def plot_residuals(
    model,
    title: str = "",
    ax: matplotlib.axes.Axes | None = None,
) -> matplotlib.axes.Axes:
    """
    Residuals vs. fitted values plot to check homoscedasticity.

    Parameters
    ----------
    model : RegressionResult
        Returned by any ``engstats`` regression function.
    title : str
    ax : matplotlib.axes.Axes, optional
    """
    if ax is None:
        _, ax = plt.subplots()
    ax.scatter(model.fitted_values, model.residuals, alpha=0.6)
    ax.axhline(0, linestyle="--", linewidth=1.5, color="grey")
    ax.set_title(title or "Residuals vs. Fitted Values")
    ax.set_xlabel("Fitted values")
    ax.set_ylabel("Residuals")
    plt.tight_layout()
    return ax


def plot_qq(
    model,
    title: str = "",
    ax: matplotlib.axes.Axes | None = None,
) -> matplotlib.axes.Axes:
    """
    Normal Q-Q plot of regression residuals.

    Parameters
    ----------
    model : RegressionResult
    title : str
    ax : matplotlib.axes.Axes, optional
    """
    if ax is None:
        _, ax = plt.subplots()
    (osm, osr), (slope, intercept, _) = stats.probplot(model.residuals)
    ax.scatter(osm, osr, alpha=0.7, zorder=3)
    line_x = np.array([min(osm), max(osm)])
    ax.plot(line_x, slope * line_x + intercept, "r--", linewidth=1.8, label="Normal line")
    ax.set_title(title or "Normal Q-Q Plot")
    ax.set_xlabel("Theoretical quantiles")
    ax.set_ylabel("Sample quantiles")
    ax.legend()
    plt.tight_layout()
    return ax
