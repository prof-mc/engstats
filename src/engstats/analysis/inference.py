"""
engstats.analysis.inference
============================
Hypothesis testing functions wrapping scipy.stats with clean output.
"""

import numpy as np
import pandas as pd
from scipy import stats
from engstats.utils.validation import require_numeric_1d


class InferenceResult:
    """Container for hypothesis test output."""

    def __init__(self, test_name, statistic, p_value, extra=None):
        self.test_name = test_name
        self.statistic = statistic
        self.p_value = p_value
        self._extra = extra or {}

    def __repr__(self):
        sig = "significant" if self.p_value < 0.05 else "not significant"
        return (
            f"{self.test_name}\n"
            f"  statistic = {self.statistic:.4f}\n"
            f"  p-value   = {self.p_value:.4f}  ({sig} at α=0.05)"
        )

    def summary(self) -> pd.DataFrame:
        row = {"test": self.test_name, "statistic": self.statistic, "p_value": self.p_value}
        row.update(self._extra)
        return pd.DataFrame([row])


def one_sample_ttest(data, popmean: float, alternative: str = "two-sided") -> InferenceResult:
    """
    One-sample t-test: test whether the sample mean equals a hypothesised value.

    Parameters
    ----------
    data : array-like
    popmean : float
        Hypothesised population mean (H₀ value).
    alternative : {'two-sided', 'less', 'greater'}

    Returns
    -------
    InferenceResult
    """
    arr = require_numeric_1d(data, "data")
    t, p = stats.ttest_1samp(arr, popmean, alternative=alternative)
    return InferenceResult("One-sample t-test", t, p, {"popmean": popmean, "alternative": alternative})


def two_sample_ttest(
    a, b, equal_var: bool = False, alternative: str = "two-sided"
) -> InferenceResult:
    """
    Independent two-sample t-test (Welch by default).

    Parameters
    ----------
    a, b : array-like
        The two groups.
    equal_var : bool
        If True, use Student's t-test (assume equal variances).
    alternative : {'two-sided', 'less', 'greater'}
    """
    a = require_numeric_1d(a, "a")
    b = require_numeric_1d(b, "b")
    t, p = stats.ttest_ind(a, b, equal_var=equal_var, alternative=alternative)
    name = "Two-sample t-test" + ("" if equal_var else " (Welch)")
    return InferenceResult(name, t, p, {"n_a": len(a), "n_b": len(b), "alternative": alternative})


def paired_ttest(before, after, alternative: str = "two-sided") -> InferenceResult:
    """
    Paired (dependent) samples t-test.

    Parameters
    ----------
    before, after : array-like
        Matched measurements (must be the same length).
    alternative : {'two-sided', 'less', 'greater'}
    """
    before = require_numeric_1d(before, "before")
    after = require_numeric_1d(after, "after")
    if len(before) != len(after):
        raise ValueError(
            f"'before' and 'after' must be the same length "
            f"(got {len(before)} and {len(after)})."
        )
    t, p = stats.ttest_rel(before, after, alternative=alternative)
    return InferenceResult("Paired t-test", t, p, {"n": len(before), "alternative": alternative})


def chi_square_test(observed, expected=None) -> InferenceResult:
    """
    Chi-square goodness-of-fit or test of independence.

    Parameters
    ----------
    observed : array-like or 2-D array
        Observed frequencies. Pass a 2-D contingency table for independence.
    expected : array-like, optional
        Expected frequencies for goodness-of-fit. Defaults to equal proportions.
    """
    obs = np.asarray(observed)
    if obs.ndim == 2:
        chi2, p, dof, _ = stats.chi2_contingency(obs)
        return InferenceResult("Chi-square test of independence", chi2, p, {"dof": dof})
    chi2, p = stats.chisquare(obs, f_exp=expected)
    return InferenceResult("Chi-square goodness-of-fit", chi2, p)


def anova_oneway(*groups) -> InferenceResult:
    """
    One-way ANOVA: test whether ≥3 group means are equal.

    Parameters
    ----------
    *groups : array-like
        Two or more groups of numeric data.

    Examples
    --------
    >>> es.anova_oneway(group_a, group_b, group_c)
    """
    cleaned = [require_numeric_1d(g, f"group_{i+1}") for i, g in enumerate(groups)]
    f, p = stats.f_oneway(*cleaned)
    return InferenceResult("One-way ANOVA", f, p, {"k": len(cleaned)})
