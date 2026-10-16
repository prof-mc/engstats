"""
engstats.plots.inference
==========================
Visualisations of hypothesis-test results.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.axes


# Default x-axis label for the parameter each t-test's interval covers
_PARAM_LABELS = {
    "One-sample": "μ",
    "Two-sample": "μ_a − μ_b",
    "Paired": "μ_d",
}


def plot_confidence_interval(
    result,
    param_name: str = "",
    title: str = "",
    ax: matplotlib.axes.Axes | None = None,
) -> matplotlib.axes.Axes:
    """
    Visualise the confidence interval from a t-test result.

    A dashed grey line marks the null-hypothesis value (``popmean`` for a
    one-sample test, 0 otherwise), so you can see at a glance whether the
    interval excludes it. A two-sided interval is drawn as an error bar
    around its centre; a one-sided interval is drawn from its finite bound
    with an arrow pointing towards the open side.

    Parameters
    ----------
    result : InferenceResult
        Returned by ``one_sample_ttest``, ``two_sample_ttest`` or
        ``paired_ttest``.
    param_name : str
        X-axis label. Defaults to the parameter the interval covers
        (``μ``, ``μ_a − μ_b`` or ``μ_d``).
    title : str
        Plot title. Defaults to ``'<confidence>% Confidence Interval'``.
    ax : matplotlib.axes.Axes, optional
        Existing axes to draw on.

    Returns
    -------
    matplotlib.axes.Axes

    Examples
    --------
    >>> df = es.load_dataset("circuits")
    >>> result = es.one_sample_ttest(df["measured_ohm"], popmean=100)
    >>> es.plot_confidence_interval(result)
    """
    lo, hi = result.conf_int_low, result.conf_int_high
    if lo is None or hi is None:
        raise ValueError(
            f"'{result.test_name}' has no confidence interval. "
            f"Only t-test results can be plotted."
        )

    null_value = result._extra.get("popmean", 0)
    confidence = result._extra.get("confidence")

    if ax is None:
        _, ax = plt.subplots(figsize=(6, 2.5))

    if np.isfinite(lo) and np.isfinite(hi):
        est = (lo + hi) / 2
        ax.errorbar(est, 0, xerr=[[est - lo], [hi - est]], fmt="o",
                    capsize=8, capthick=2, linewidth=2, zorder=3)
        points = [lo, hi, null_value]
    else:
        # One-sided: one bound is infinite, so draw the finite bound with an
        # arrow running off towards the open side
        bound = lo if np.isfinite(lo) else hi
        direction = 1 if np.isfinite(lo) else -1
        span = max(abs(bound - null_value), abs(bound) * 0.05, 1e-9)
        tip = bound + direction * 1.5 * span
        ax.annotate("", xy=(tip, 0), xytext=(bound, 0),
                    arrowprops=dict(arrowstyle="-|>", color="C0", linewidth=2))
        ax.plot([bound, bound], [-0.15, 0.15], color="C0", linewidth=2)
        points = [bound, tip, null_value]

    ax.axvline(null_value, linestyle="--", color="grey", linewidth=1,
               label=f"H₀ value = {null_value:g}")

    pad = 0.15 * (max(points) - min(points) or 1)
    ax.set_xlim(min(points) - pad, max(points) + pad)
    ax.set_ylim(-1, 1)
    ax.set_yticks([])

    default_label = next((v for k, v in _PARAM_LABELS.items()
                          if result.test_name.startswith(k)), "Parameter")
    ax.set_xlabel(param_name or default_label)
    ax.set_title(title or (f"{confidence * 100:g}% Confidence Interval"
                           if confidence else "Confidence Interval"))
    ax.legend(loc="upper right")
    plt.tight_layout()
    return ax
