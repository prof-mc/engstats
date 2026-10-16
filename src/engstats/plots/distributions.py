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
        Numeric data.
    bins : int or str
        Number of bins, or a NumPy binning rule such as ``'auto'``,
        ``'sturges'`` or ``'fd'``. Default ``'auto'``.
    title : str
        Plot title. Defaults to ``"Histogram"``.
    xlabel : str
        X-axis label. Default ``"Value"``.
    kde : bool
        Overlay a kernel density estimate. Default False.
    color : str, optional
        Bar colour.
    ax : matplotlib.axes.Axes, optional
        Existing axes to draw on.

    Returns
    -------
    matplotlib.axes.Axes

    Examples
    --------
    >>> df = es.load_dataset("concrete")
    >>> es.plot_histogram(df["strength_mpa"], bins=10, kde=True, xlabel="Strength (MPa)")
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
        Numeric data.
    title : str
        Plot title. Defaults to ``"Density Estimate"``.
    xlabel : str
        X-axis label. Default ``"Value"``.
    fill : bool
        Shade the area under the curve. Default True.
    color : str, optional
        Curve colour.
    ax : matplotlib.axes.Axes, optional
        Existing axes to draw on.

    Returns
    -------
    matplotlib.axes.Axes

    Examples
    --------
    >>> df = es.load_dataset("concrete")
    >>> es.plot_kde(df["strength_mpa"], xlabel="Strength (MPa)")
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

        es.plot_boxplot(df, x="material", y="load_capacity_kn")

    Points beyond 1.5 × IQR from the box are drawn individually as outliers.

    Parameters
    ----------
    data : pd.DataFrame, optional
        Source DataFrame (usage B).
    x : str, optional
        Categorical column; one box per group. Omit for a single box
        of ``y``.
    y : str, optional
        Numeric column (usage B).
    values : array-like, optional
        Numeric data (usage A). Takes precedence over ``data``.
    title : str
        Plot title. Defaults to ``"Box Plot"``.
    ax : matplotlib.axes.Axes, optional
        Existing axes to draw on.

    Returns
    -------
    matplotlib.axes.Axes

    Raises
    ------
    ValueError
        If neither ``values`` nor both ``data`` and ``y`` are given.
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

    Usage mirrors [`plot_boxplot`][engstats.plots.distributions.plot_boxplot]: pass ``values=`` for one array, or
    ``data``, ``y`` and optionally ``x`` for grouped violins.

    Parameters
    ----------
    data : pd.DataFrame, optional
        Source DataFrame (usage B).
    x : str, optional
        Categorical column; one violin per group. Omit for a single violin
        of ``y``.
    y : str, optional
        Numeric column (usage B).
    values : array-like, optional
        Numeric data (usage A). Takes precedence over ``data``.
    title : str
        Plot title. Defaults to ``"Violin Plot"``.
    ax : matplotlib.axes.Axes, optional
        Existing axes to draw on.

    Returns
    -------
    matplotlib.axes.Axes

    Raises
    ------
    ValueError
        If neither ``values`` nor both ``data`` and ``y`` are given.

    Examples
    --------
    >>> df = es.load_dataset("bridges")
    >>> es.plot_violin(df, x="material", y="load_capacity_kn")
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

    The ECDF at value *v* is the proportion of observations ≤ *v*.

    Parameters
    ----------
    data : array-like
        Numeric data.
    title : str
        Plot title. Defaults to ``"ECDF"``.
    xlabel : str
        X-axis label. Default ``"Value"``.
    color : str, optional
        Line colour.
    ax : matplotlib.axes.Axes, optional
        Existing axes to draw on.

    Returns
    -------
    matplotlib.axes.Axes

    Examples
    --------
    >>> df = es.load_dataset("concrete")
    >>> es.plot_ecdf(df["strength_mpa"], xlabel="Strength (MPa)")
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


def plot_probability(
    data: pd.DataFrame,
    x: str,
    ci: float = 0.95,
    xlim_left: float | None = None,
    title: str = "",
    xlabel: str = "",
    ax: matplotlib.axes.Axes | None = None,
) -> matplotlib.axes.Axes:
    """
    Normal probability plot with a fitted line and prediction interval band.

    The ordered data are plotted against their normal scores. The y-axis is
    labelled in percentiles on a probit scale, so it looks like normal
    probability paper: points close to a straight line suggest the data are
    approximately normal.

    Parameters
    ----------
    data : pd.DataFrame
        DataFrame containing the variable of interest.
    x : str
        Column name of the variable to assess.
    ci : float
        Confidence level for the prediction interval (0–1). Default 0.95.
    xlim_left : float, optional
        Force the left limit of the x-axis. Useful for variables with a
        natural lower bound (e.g. pass ``xlim_left=0`` for time or load).
        Defaults to None (let matplotlib decide).
    title : str
        Plot title. Defaults to ``'Normal Probability Plot: <x>'``.
    xlabel : str
        X-axis label. Defaults to the column name.
    ax : matplotlib.axes.Axes, optional
        Existing axes to draw on.

    Returns
    -------
    matplotlib.axes.Axes

    Examples
    --------
    >>> df = es.load_dataset("concrete")
    >>> es.plot_probability(df, x="strength_mpa")
    >>> es.plot_probability(df, x="age_days", ci=0.99, xlim_left=0)
    """
    import statsmodels.api as sm
    from matplotlib.lines import Line2D
    from scipy import stats

    from engstats.utils.validation import require_dataframe
    require_dataframe(data, columns=[x])

    if not (0 < ci < 1):
        raise ValueError(f"'ci' must be between 0 and 1, got {ci}.")

    values = pd.to_numeric(data[x], errors="coerce").dropna().to_numpy()
    if len(values) < 3:
        raise ValueError(
            f"Column '{x}' has fewer than 3 numeric, non-null values — "
            f"not enough data to fit a probability plot."
        )

    # Normal scores (osm) and ordered data (osr)
    (osm, osr), _ = stats.probplot(values, dist="norm")

    # Regress normal scores on the ordered data, then predict over a grid
    ols_model = sm.OLS(osm, sm.add_constant(osr)).fit()
    x_grid = np.linspace(osr.min(), osr.max(), 300)
    sf = ols_model.get_prediction(sm.add_constant(x_grid)).summary_frame(alpha=1 - ci)

    if ax is None:
        _, ax = plt.subplots(figsize=(6, 6))

    ax.scatter(osr, osm, s=18, alpha=0.7, zorder=3, label="Data")
    ax.plot(x_grid, sf["mean"], "r--", linewidth=2, label="Fitted line")
    ax.fill_between(
        x_grid,
        sf["obs_ci_lower"],
        sf["obs_ci_upper"],
        color="green",
        alpha=0.15,
        label=f"{ci * 100:g}% PI",
    )

    # Percentile ticks on the probit scale
    pticks = np.array([1, 5, 10, 25, 50, 75, 90, 95, 99])
    ax.set_yticks(stats.norm.ppf(pticks / 100))
    ax.set_yticklabels([f"{p:g}%" for p in pticks])

    if xlim_left is not None:
        ax.set_xlim(left=xlim_left)

    handles, _ = ax.get_legend_handles_labels()
    handles.append(Line2D(
        [], [], linestyle="none",
        label=f"n = {len(values)}\nμ = {values.mean():.3g}\nσ = {values.std(ddof=1):.3g}",
    ))

    ax.set_xlabel(xlabel or x)
    ax.set_ylabel("Percentile")
    ax.set_title(title or f"Normal Probability Plot: {x}")
    ax.legend(handles=handles)
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

    Bars show the count of each category in descending order; the dashed
    line (right axis) shows the cumulative percentage. Missing values are
    dropped.

    Parameters
    ----------
    data : pd.DataFrame
        Source DataFrame.
    x : str
        Categorical column to summarize.
    title : str
        Plot title.
    xlabel : str
        X-axis label.
    ax : matplotlib.axes.Axes, optional
        Existing axes to draw on.

    Returns
    -------
    matplotlib.axes.Axes
        The bar-chart axes. The cumulative line is on a twin axes.

    Raises
    ------
    ValueError
        If ``x`` is not a column or has no non-missing values.

    Examples
    --------
    >>> df = es.load_dataset("bridges")
    >>> es.plot_pareto(df, x="material")
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
    """
    Compare several numeric columns side by side on one shared y-axis.

    Parameters
    ----------
    data : pd.DataFrame
        Source DataFrame.
    x : list of str or str
        Column(s) to compare, one position on the x-axis each. Names not
        found in ``data`` are skipped silently.
    style : {'individual', 'box'}
        ``'individual'`` draws every observation as a dot (swarm plot);
        ``'box'`` draws a box plot per column. Default ``'individual'``.
    ax : matplotlib.axes.Axes, optional
        Existing axes to draw on. A new 8 × 4 figure is created otherwise.

    Returns
    -------
    matplotlib.axes.Axes

    Raises
    ------
    ValueError
        If ``style`` is not ``'individual'`` or ``'box'``.

    Examples
    --------
    >>> df = es.load_dataset("circuits")
    >>> es.plot_side_by_side(df, x=["nominal_ohm", "measured_ohm"], style="box")
    """
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
