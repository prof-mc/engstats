"""
engstats.analysis.inference
============================
Hypothesis testing functions wrapping scipy.stats with clean output.
"""

import numpy as np
import pandas as pd
from scipy import stats
from engstats.utils.validation import require_numeric_1d
pd.set_option('display.precision', 4)

class InferenceResult:
    """Container for hypothesis test output."""

    def __init__(self, test_name, statistic, p_value, conf_int_low=None, conf_int_high=None, extra=None):
        self.test_name = test_name
        self.statistic = statistic
        self.p_value = p_value
        self.conf_int_low = conf_int_low
        self.conf_int_high = conf_int_high
        self._extra = extra or {}

    def __repr__(self):
        pd.set_option("display.precision", 4)
        repr_str = (
            f"{self.test_name}\n"
            f"  statistic = {self.statistic}\n"
            f"  p-value   = {self.p_value}\n"
        )
        if 't-test' in self.test_name:
            repr_str += (
                f"  CI_low  = {self.conf_int_low}\t"
                f"  CI_high = {self.conf_int_high}\n"
            )
        return repr_str


    def summary(self) -> pd.DataFrame:
        pd.set_option("display.precision", 4)
        row = {"test": self.test_name,
               "statistic": self.statistic,
               "p_value": self.p_value,
               }
        if 't-test' in self.test_name:
            row["CI_low"] = self.conf_int_low
            row["CI_high"] = self.conf_int_high
        row.update(self._extra)
        return pd.DataFrame([row])


def one_sample_ttest(data, popmean: float, alternative: str = "two-sided",
                     confidence=0.95) -> InferenceResult:
    """
    One-sample t-test: test whether the sample mean equals a hypothesised value.

    Parameters
    ----------
    data : array-like
    popmean : float
        Null hypothesis population mean (H₀ value).
    alternative : {'two-sided', 'less', 'greater'}
    confidence : float
        Confidence level for calculating confidence interval.

    Returns
    -------
    InferenceResult
    """
    arr = require_numeric_1d(data, "data")
    result = stats.ttest_1samp(arr, popmean, alternative=alternative)

    t = result.statistic
    p = result.pvalue
    conf_int_low = result.confidence_interval(confidence).low
    conf_int_high = result.confidence_interval(confidence).high


    return InferenceResult("One-sample t-test", t, p, conf_int_low, conf_int_high,
                           {"popmean": popmean, "n": len(arr),
                            "alternative": alternative, "confidence": confidence})


def two_sample_ttest(
    a, b, equal_var: bool = False, alternative: str = "two-sided", confidence=0.95
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
    confidence : float
        Confidence level for calculating confidence interval.
    """
    a = require_numeric_1d(a, "a")
    b = require_numeric_1d(b, "b")

    result = stats.ttest_ind(a, b, equal_var=equal_var, alternative=alternative)

    t = result.statistic
    p = result.pvalue
    conf_int_low = result.confidence_interval(confidence).low
    conf_int_high = result.confidence_interval(confidence).high

    name = "Two-sample t-test" + ("" if equal_var else " (Welch)")
    return InferenceResult(name, t, p, conf_int_low, conf_int_high,
                           {"n_a": len(a), "n_b": len(b), "alternative": alternative, "confidence": confidence})


def paired_ttest(before, after, alternative: str = "two-sided", confidence=0.95) -> InferenceResult:
    """
    Paired (dependent) samples t-test.

    Parameters
    ----------
    before, after : array-like
        Matched measurements (must be the same length).
    alternative : {'two-sided', 'less', 'greater'}
    confidence : float
        Confidence level for calculating confidence interval.
    """
    before = require_numeric_1d(before, "before")
    after = require_numeric_1d(after, "after")
    if len(before) != len(after):
        raise ValueError(
            f"'before' and 'after' must be the same length "
            f"(got {len(before)} and {len(after)})."
        )
    result = stats.ttest_rel(before, after, alternative=alternative)

    t = result.statistic
    p = result.pvalue
    conf_int_low = result.confidence_interval(confidence).low
    conf_int_high = result.confidence_interval(confidence).high

    return InferenceResult("Paired t-test", t, p, conf_int_low, conf_int_high,
                           {"n": len(before), "alternative": alternative, "confidence": confidence})


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
