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

import matplotlib.colors as mcolors

def plot_scatter(
    data: pd.DataFrame,
    x: str,
    y: str,
    group: str = None,
    title: str = "",
    ax: matplotlib.axes.Axes | None = None,
) -> matplotlib.axes.Axes:
    """
    Scatter plot of ``y`` against ``x``, optionally split by group.

    Parameters
    ----------
    data : pd.DataFrame
        Source DataFrame (not modified).
    x, y : str
        Column names for the horizontal and vertical axes.
    group : str, optional
        Categorical column; each level gets its own colour and marker. If
        every level is a valid Matplotlib colour name (e.g. ``"red"``,
        ``"tab:blue"``), those colours are used directly.
    title : str
        Plot title. Defaults to ``"Scatter Plot: <y> vs <x>"``.
    ax : matplotlib.axes.Axes, optional
        Existing axes to draw on.

    Returns
    -------
    matplotlib.axes.Axes

    For a fitted line through the points, use
    [`plot_scatter_regression`][engstats.plots.regression.plot_scatter_regression].

    Examples
    --------
    >>> df = es.load_dataset("bridges")
    >>> es.plot_scatter(df, x="span_m", y="load_capacity_kn", group="material")
    """
    if ax is None:
        _, ax = plt.subplots()

    def is_color(val):
        try:
            mcolors.to_rgba(val)
            return True
        except ValueError:
            return False

    if group is not None:
        data = data.copy()  # don't change the caller's column dtype
        data[group] = data[group].astype("category")
        unique_vals = data[group].dropna().unique()

        # Check if all values are valid colors
        if all(is_color(val) for val in unique_vals):
            palette = {val: val for val in unique_vals}
        else:
            palette = None  # fallback to seaborn default

        sns.scatterplot(
            data=data,
            x=x,
            y=y,
            hue=group,
            style=group,
            palette=palette,
            ax=ax
        )
    else:
        sns.scatterplot(data=data, x=x, y=y, ax=ax)

    ax.set_title(title or f"Scatter Plot: {y} vs {x}")
    ax.set_xlabel(x)
    ax.set_ylabel(y)
    plt.tight_layout()
    return ax

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

    The band is a confidence interval for the **mean** response, estimated
    by bootstrap. It is not a prediction interval for new observations.

    Parameters
    ----------
    data : pd.DataFrame
        Source DataFrame.
    x, y : str
        Predictor and response column names.
    degree : int
        Polynomial degree of the fitted curve. 1 = straight line. Default 1.
    ci : int or None
        Confidence level for the band, in percent (0–100). ``None`` hides
        the band. Default 95.
    title : str
        Plot title. Defaults to ``"Scatter Plot: <y> vs <x>"``.
    ax : matplotlib.axes.Axes, optional
        Existing axes to draw on.

    Returns
    -------
    matplotlib.axes.Axes

    Examples
    --------
    >>> df = es.load_dataset("concrete")
    >>> es.plot_scatter_regression(df, x="water_cement", y="strength_mpa")
    >>> es.plot_scatter_regression(df, x="water_cement", y="strength_mpa", degree=2, ci=None)
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

    A good fit shows points scattered evenly around zero with constant
    spread. A funnel shape suggests non-constant variance; a curve suggests
    a missing term.

    Parameters
    ----------
    model : RegressionResult
        Returned by any ``engstats`` regression function.
    title : str
        Plot title. Defaults to ``"Residuals vs. Fitted Values"``.
    ax : matplotlib.axes.Axes, optional
        Existing axes to draw on.

    Returns
    -------
    matplotlib.axes.Axes

    Examples
    --------
    >>> df = es.load_dataset("concrete")
    >>> model = es.simple_linear_regression(df, x="water_cement", y="strength_mpa")
    >>> es.plot_residuals(model)
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

    Points close to the dashed reference line indicate approximately normal
    errors. Systematic curvature at the ends suggests skew or heavy tails.

    Parameters
    ----------
    model : RegressionResult
        Returned by any ``engstats`` regression function.
    title : str
        Plot title. Defaults to ``"Normal Q-Q Plot"``.
    ax : matplotlib.axes.Axes, optional
        Existing axes to draw on.

    Returns
    -------
    matplotlib.axes.Axes

    Examples
    --------
    >>> df = es.load_dataset("concrete")
    >>> model = es.simple_linear_regression(df, x="water_cement", y="strength_mpa")
    >>> es.plot_qq(model)
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


def plot_runplot(
    data: pd.DataFrame,
    x: str,
    y: list[str],
    separate: bool = True,
    title: str = "",
) -> list[matplotlib.axes.Axes]:
    """
    Plot one to four variables in run order, on shared or separate axes.

    Use to look for trends, shifts or cycles over time or observation order.
    Always creates a new figure.

    Parameters
    ----------
    data : pd.DataFrame
        DataFrame containing all variables as columns.
    x : str
        Column name for the x-axis (e.g. time, year, observation order).
    y : list of str
        Column name(s) for the y-axis. Must contain between 1 and 4 names.
    separate : bool
        If True, each variable gets its own subplot with a shared x-axis.
        If False, all series are overlaid on a single axes with a legend.
        Default True.
    title : str
        Overall figure title.

    Returns
    -------
    list of matplotlib.axes.Axes
        One Axes per variable when ``separate=True``, a single-element list
        otherwise. Always a list.

    Raises
    ------
    ValueError
        If ``y`` is not a list of 1 to 4 column names.
    KeyError
        If ``x`` or any name in ``y`` is not a column of ``data``.

    Examples
    --------
    >>> df = es.load_dataset("circuits").reset_index()
    >>> es.plot_runplot(df, x="index", y=["measured_ohm"])
    >>> es.plot_runplot(df, x="index", y=["nominal_ohm", "measured_ohm"], separate=False)
    """
    if not isinstance(y, list) or not (1 <= len(y) <= 4):
        raise ValueError(
            f"'y' must be a list of 1 to 4 column names, "
            f"got {len(y) if isinstance(y, list) else type(y).__name__}."
        )
    from engstats.utils.validation import require_dataframe
    from matplotlib.ticker import MaxNLocator
    require_dataframe(data, columns=[x] + y)

    n = len(y)

    if not separate:
        fig, ax_single = plt.subplots(figsize=(9, 4))
        for yvar in y:
            ax_single.plot(data[x], data[yvar], marker="o", label=yvar)
        ax_single.set_xlabel(x)
        ax_single.set_ylabel("Value")
        ax_single.set_title(title or "Time Series")
        ax_single.xaxis.set_major_locator(MaxNLocator(integer=True))
        ax_single.legend()
        plt.tight_layout()
        return [ax_single]

    fig, axes = plt.subplots(n, 1, figsize=(9, 3 * n), sharex=True)

    axes = list(np.atleast_1d(axes))

    for ax, yvar in zip(axes, y):
        ax.plot(data[x], data[yvar], marker="o", label=yvar)
        ax.set_ylabel(yvar)
        # ax.legend(loc="upper right")
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    axes[-1].set_xlabel(x)

    if title:
        fig.suptitle(title, y=1.01)

    plt.tight_layout()
    return axes


def plot_runplot_split(
    data: pd.DataFrame,
    x: str,
    y: str,
    group: str,
    ax: matplotlib.axes.Axes = None
):
    """
    Run plot of one variable, with a separate line for each group.

    Groups keep the order in which they first appear in ``data``. Useful for
    comparing batches, shifts or machines over the same run order.

    Parameters
    ----------
    data : pd.DataFrame
        Source DataFrame (not modified).
    x : str
        Column for the x-axis (time or observation order).
    y : str
        Numeric column to plot.
    group : str or None
        Categorical column; each level gets its own line colour and marker.
        ``None`` draws a single line.
        If every level is a valid Matplotlib colour name, those colours are
        used directly.
    ax : matplotlib.axes.Axes, optional
        Existing axes to draw on.

    Returns
    -------
    matplotlib.axes.Axes

    Examples
    --------
    >>> df = es.load_dataset("circuits").reset_index()
    >>> es.plot_runplot_split(df, x="index", y="measured_ohm", group="batch")
    """
    if ax is None:
        _, ax = plt.subplots()

    def is_color(val):
        try:
            mcolors.to_rgba(val)
            return True
        except ValueError:
            return False

    if group is not None:
        plot_data = data.copy()
        group_order = pd.unique(plot_data[group].dropna())


        plot_data[group] = pd.Categorical(
            plot_data[group],
            categories=group_order,
            ordered=True,
        )

        # Check if all values are valid colors
        if all(is_color(val) for val in group_order):
            palette = {val: val for val in group_order}
        else:
            palette = None  # fallback to seaborn default

        sns.lineplot(
            data=plot_data,
            x=x,
            y=y,
            hue=group,
            style=group,
            hue_order=group_order,
            style_order=group_order,
            palette=palette,
            markers=True,
            ax=ax
        )
    else:
        sns.lineplot(data=data, x=x, y=y, marker="o", ax=ax)

    return ax