"""
engstats — Engineering Statistics Companion Package
====================================================
A student-friendly wrapper around statsmodels, scipy, seaborn, and matplotlib
for use in freshman engineering statistics courses.

Quick start
-----------
>>> import engstats as es
>>> df = es.load_dataset("concrete")
>>> model = es.simple_linear_regression(df, x="water_cement", y="strength")
>>> es.plot_scatter_regression(df, x="water_cement", y="strength")
"""

# Apply the default plot theme on import
from engstats.plots.theme import apply_theme
apply_theme()

# --- Analysis ---
from engstats.analysis.descriptive import (
    five_number_summary,
    summary_stats,
)
from engstats.analysis.regression import (
    simple_linear_regression,
    multiple_linear_regression,
    polynomial_regression,
)
from engstats.analysis.inference import (
    one_sample_ttest,
    two_sample_ttest,
    paired_ttest,
    chi_square_test,
    anova_oneway,
)
from engstats.analysis.probability import (
    normal_prob,
    binomial_prob,
    poisson_prob,
)
from engstats.analysis.correlation import (
    pearson_r,
    spearman_rho,
    correlation_matrix,
)

# --- Plots ---
from engstats.plots.distributions import (
    plot_histogram,
    plot_kde,
    plot_boxplot,
    plot_violin,
    plot_ecdf,
    plot_pareto,
    plot_side_by_side
)
from engstats.plots.regression import (
    plot_scatter,
    plot_scatter_regression,
    plot_residuals,
    plot_qq,
    plot_runplot,
    plot_runplot_split
)
from engstats.plots.categorical import (
    plot_bar,
    plot_grouped_bar,
    plot_stackeddot
)
from engstats.plots.probability import (
    plot_normal_curve,
    plot_binomial_pmf,
    plot_confidence_interval,
    plot_probability_order,
)
from engstats.plots.multivariate import (
    plot_correlation_heatmap,
    plot_pairplot,
)

# --- Utilities ---
from engstats.utils.io import load_dataset

__all__ = [
    # descriptive
    "five_number_summary", "summary_stats",
    # regression analysis
    "simple_linear_regression", "multiple_linear_regression", "polynomial_regression",
    # inference
    "one_sample_ttest", "two_sample_ttest", "paired_ttest",
    "chi_square_test", "anova_oneway",
    # probability
    "normal_prob", "binomial_prob", "poisson_prob",
    # correlation
    "pearson_r", "spearman_rho", "correlation_matrix",
    # distribution plots
    "plot_histogram", "plot_kde", "plot_boxplot", "plot_violin", "plot_ecdf",
    # regression plots
    "plot_scatter_regression", "plot_residuals", "plot_qq",
    # categorical plots
    "plot_bar", "plot_grouped_bar",
    # probability plots
    "plot_normal_curve", "plot_binomial_pmf", "plot_confidence_interval",
    # multivariate plots
    "plot_correlation_heatmap", "plot_pairplot",
    # io
    "load_dataset",
]
