"""
engstats.plots.probability
============================
Visualisations for common probability distributions.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.axes
from scipy import stats


def plot_normal_curve(
    mean: float = 0,
    std: float = 1,
    shade_below: float | None = None,
    shade_above: float | None = None,
    title: str = "",
    ax: matplotlib.axes.Axes | None = None,
) -> matplotlib.axes.Axes:
    """
    Plot a normal distribution curve with optional shaded tail area.

    Parameters
    ----------
    mean, std : float
        Distribution parameters.
    shade_below : float, optional
        Shade P(X ≤ shade_below).
    shade_above : float, optional
        Shade P(X > shade_above).
    title : str
    ax : matplotlib.axes.Axes, optional

    Examples
    --------
    >>> es.plot_normal_curve(mean=70, std=5, shade_above=80)
    """
    if std <= 0:
        raise ValueError("std must be positive.")
    if ax is None:
        _, ax = plt.subplots()
    lo, hi = mean - 4 * std, mean + 4 * std
    x = np.linspace(lo, hi, 400)
    y = stats.norm.pdf(x, loc=mean, scale=std)
    ax.plot(x, y, linewidth=2)
    if shade_below is not None:
        mask = x <= shade_below
        ax.fill_between(x[mask], y[mask], alpha=0.35, label=f"P(X ≤ {shade_below:.2f})")
    if shade_above is not None:
        mask = x >= shade_above
        ax.fill_between(x[mask], y[mask], alpha=0.35, label=f"P(X > {shade_above:.2f})")
    ax.set_title(title or f"Normal Distribution (μ={mean}, σ={std})")
    ax.set_xlabel("x")
    ax.set_ylabel("Density")
    if shade_below is not None or shade_above is not None:
        ax.legend()
    plt.tight_layout()
    return ax


def plot_binomial_pmf(
    n: int,
    p: float,
    highlight_k: int | None = None,
    title: str = "",
    ax: matplotlib.axes.Axes | None = None,
) -> matplotlib.axes.Axes:
    """
    Bar chart of the Binomial(n, p) probability mass function.

    Parameters
    ----------
    n : int
        Number of trials.
    p : float
        Success probability.
    highlight_k : int, optional
        Highlight a specific k value in a contrasting colour.
    title : str
    ax : matplotlib.axes.Axes, optional
    """
    if ax is None:
        _, ax = plt.subplots()
    ks = np.arange(0, n + 1)
    probs = stats.binom.pmf(ks, n=n, p=p)
    colors = ["#DC2626" if highlight_k is not None and k == highlight_k else "#2563EB" for k in ks]
    ax.bar(ks, probs, color=colors, edgecolor="white", linewidth=0.5)
    ax.set_title(title or f"Binomial PMF (n={n}, p={p})")
    ax.set_xlabel("k (number of successes)")
    ax.set_ylabel("P(X = k)")
    plt.tight_layout()
    return ax


def plot_confidence_interval(
    result,
    param_name: str = "mean",
    title: str = "",
    ax: matplotlib.axes.Axes | None = None,
) -> matplotlib.axes.Axes:
    """
    Visualise a confidence interval from a hypothesis-test result.

    Parameters
    ----------
    result : InferenceResult
        Returned by any ``engstats`` inference function.
    param_name : str
        Label for the x-axis.
    title : str
    ax : matplotlib.axes.Axes, optional
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(6, 2.5))
    # Try to extract CI from the result's _extra dict or fall back to t-based CI
    ci = result._extra.get("ci", None)
    if ci is None:
        ax.text(0.5, 0.5, "No CI available for this test type.",
                ha="center", va="center", transform=ax.transAxes)
        return ax
    lo, hi, est = ci
    ax.errorbar(est, 0, xerr=[[est - lo], [hi - est]], fmt="o",
                capsize=8, capthick=2, linewidth=2)
    ax.axvline(0, linestyle="--", color="grey", linewidth=1)
    ax.set_yticks([])
    ax.set_xlabel(param_name)
    ax.set_title(title or "Confidence Interval")
    plt.tight_layout()
    return ax
