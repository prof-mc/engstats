"""Smoke tests for plot functions — verify they run and return Axes."""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")   # non-interactive backend for CI
import matplotlib.pyplot as plt
import pytest

from engstats.plots.distributions import (
    plot_histogram, plot_kde, plot_boxplot, plot_violin, plot_ecdf
)
from engstats.plots.regression import plot_scatter_regression, plot_residuals, plot_qq
from engstats.plots.categorical import plot_bar, plot_grouped_bar
from engstats.plots.multivariate import plot_correlation_heatmap, plot_pairplot
from engstats.analysis.regression import simple_linear_regression

rng = np.random.default_rng(7)
DATA = rng.normal(50, 10, 60)
DF = pd.DataFrame({
    "x": rng.uniform(0, 10, 40),
    "y": rng.normal(5, 2, 40),
    "group": rng.choice(["A", "B"], 40),
})


def teardown_function():
    plt.close("all")


def test_histogram_returns_axes():
    ax = plot_histogram(DATA)
    assert hasattr(ax, "get_title")


def test_kde_returns_axes():
    ax = plot_kde(DATA)
    assert hasattr(ax, "get_title")


def test_boxplot_values():
    ax = plot_boxplot(values=DATA)
    assert hasattr(ax, "get_title")


def test_violin_values():
    ax = plot_violin(values=DATA)
    assert hasattr(ax, "get_title")


def test_ecdf_returns_axes():
    ax = plot_ecdf(DATA)
    assert hasattr(ax, "get_title")


def test_scatter_regression_returns_axes():
    ax = plot_scatter_regression(DF, x="x", y="y")
    assert hasattr(ax, "get_title")


def test_plot_residuals():
    model = simple_linear_regression(DF, x="x", y="y")
    ax = plot_residuals(model)
    assert hasattr(ax, "get_title")


def test_plot_qq():
    model = simple_linear_regression(DF, x="x", y="y")
    ax = plot_qq(model)
    assert hasattr(ax, "get_title")


def test_plot_bar():
    ax = plot_bar(DF, x="group", y="y")
    assert hasattr(ax, "get_title")


def test_plot_grouped_bar():
    DF2 = DF.copy()
    DF2["group2"] = rng.choice(["X", "Y"], len(DF))
    ax = plot_grouped_bar(DF2, x="group", y="y", hue="group2")
    assert hasattr(ax, "get_title")


def test_correlation_heatmap():
    ax = plot_correlation_heatmap(DF[["x", "y"]])
    assert hasattr(ax, "get_title")


from engstats.plots.regression import plot_scatter, plot_runplot, plot_runplot_split
from engstats.plots.categorical import plot_stackeddot

RUN = pd.DataFrame({
    "t": np.arange(12),
    "a": rng.normal(0, 1, 12),
    "b": rng.normal(0, 1, 12),
    "batch": ["A"] * 6 + ["B"] * 6,
})


@pytest.mark.parametrize("cols", [["a"], ["a", "b"]])
def test_runplot_separate_returns_list_of_axes(cols):
    axes = plot_runplot(RUN, x="t", y=cols)
    assert isinstance(axes, list) and len(axes) == len(cols)
    assert all(hasattr(ax, "get_title") for ax in axes)


def test_runplot_overlay_has_legend():
    (ax,) = plot_runplot(RUN, x="t", y=["a", "b"], separate=False)
    assert ax.get_legend() is not None


def test_runplot_split_draws_on_given_axes():
    _, given = plt.subplots()
    ax = plot_runplot_split(RUN, x="t", y="a", group="batch", ax=given)
    assert ax is given and len(ax.lines) > 0


def test_runplot_split_without_group_uses_given_axes():
    _, given = plt.subplots()
    plt.subplots()  # make a different figure current
    ax = plot_runplot_split(RUN, x="t", y="a", group=None, ax=given)
    assert ax is given and len(ax.lines) > 0


def test_stackeddot_does_not_modify_input():
    df = RUN.copy()
    ax = plot_stackeddot(df, x="batch")
    assert hasattr(ax, "get_title")
    assert list(df.columns) == list(RUN.columns)


def test_scatter_group_does_not_modify_input():
    df = DF.copy()
    plot_scatter(df, x="x", y="y", group="group")
    assert df["group"].dtype == DF["group"].dtype
