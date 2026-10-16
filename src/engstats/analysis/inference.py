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
    """
    Container for hypothesis test output.

    Returned by every test in this module. Printing it shows the test name,
    statistic, p-value and (for t-tests) the confidence interval.

    Attributes
    ----------
    test_name : str
        Human-readable test name, e.g. ``"Two-sample t-test (Welch)"``.
    statistic : float
        Test statistic (t, χ², or F depending on the test).
    p_value : float
        p-value for the requested alternative.
    conf_int_low, conf_int_high : float or None
        Confidence-interval bounds. Set for t-tests only; ``None`` otherwise.
        One-sided alternatives give an infinite bound on one side.
    """

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
        """
        Return the result as a one-row DataFrame.

        Returns
        -------
        pd.DataFrame
            Columns ``test``, ``statistic``, ``p_value``, plus test-specific
            fields: t-tests add ``CI_low``, ``CI_high``, sample sizes,
            ``alternative`` and ``confidence``; chi-square tests add ``dof``;
            ANOVA adds ``k`` (number of groups) and ``n`` (total observations).
        """
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
        Numeric sample.
    popmean : float
        Null hypothesis population mean (H₀: μ = popmean).
    alternative : {'two-sided', 'less', 'greater'}
        Alternative hypothesis. ``'less'`` tests H₁: μ < popmean,
        ``'greater'`` tests H₁: μ > popmean. Default ``'two-sided'``.
    confidence : float
        Confidence level for the interval, between 0 and 1. Default 0.95.

    Returns
    -------
    InferenceResult
        ``conf_int_low`` / ``conf_int_high`` bound the population mean μ.

    Examples
    --------
    >>> df = es.load_dataset("concrete")
    >>> es.one_sample_ttest(df["strength_mpa"], popmean=30)
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
        The two independent samples.
    equal_var : bool
        If True, use the pooled (Student's) t-test, which assumes equal
        variances. Default False (Welch's t-test).
    alternative : {'two-sided', 'less', 'greater'}
        Alternative hypothesis about μ_a − μ_b. ``'less'`` tests
        H₁: μ_a < μ_b. Default ``'two-sided'``.
    confidence : float
        Confidence level for the interval, between 0 and 1. Default 0.95.

    Returns
    -------
    InferenceResult
        ``conf_int_low`` / ``conf_int_high`` bound the difference μ_a − μ_b.

    Examples
    --------
    >>> df = es.load_dataset("circuits")
    >>> a = df.loc[df["batch"] == "A", "measured_ohm"]
    >>> b = df.loc[df["batch"] == "B", "measured_ohm"]
    >>> es.two_sample_ttest(a, b)
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
        Matched measurements of equal length; element *i* of each belongs to
        the same unit.
    alternative : {'two-sided', 'less', 'greater'}
        Alternative hypothesis about the mean difference μ_d = mean(before −
        after). Default ``'two-sided'``.
    confidence : float
        Confidence level for the interval, between 0 and 1. Default 0.95.

    Returns
    -------
    InferenceResult
        ``conf_int_low`` / ``conf_int_high`` bound μ_d.

    Raises
    ------
    ValueError
        If ``before`` and ``after`` differ in length.

    Examples
    --------
    >>> df = es.load_dataset("circuits")
    >>> es.paired_ttest(df["nominal_ohm"], df["measured_ohm"])
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

    The test is chosen from the shape of ``observed``: 1-D runs a
    goodness-of-fit test, 2-D runs a test of independence on the
    contingency table.

    Parameters
    ----------
    observed : array-like or 2-D array
        Observed counts. Pass a 2-D contingency table (rows × columns) for
        the independence test.
    expected : array-like, optional
        Expected counts for goodness-of-fit, same length as ``observed`` and
        summing to the same total. Defaults to equal counts in every
        category. Ignored for 2-D input, where expected counts come from the
        row and column totals.

    Returns
    -------
    InferenceResult
        ``statistic`` is χ². ``summary()`` adds ``dof``: k − 1 for
        goodness-of-fit, (rows − 1)(columns − 1) for independence. No
        confidence interval.

    Examples
    --------
    >>> es.chi_square_test([18, 22, 20], expected=[20, 20, 20])
    >>> es.chi_square_test([[10, 20], [30, 40]])
    """
    obs = np.asarray(observed)
    if obs.ndim == 2:
        chi2, p, dof, _ = stats.chi2_contingency(obs)
        return InferenceResult("Chi-square test of independence", chi2, p,
                               extra={"dof": int(dof)})
    chi2, p = stats.chisquare(obs, f_exp=expected)
    return InferenceResult("Chi-square goodness-of-fit", chi2, p,
                           extra={"dof": obs.size - 1})


def anova_oneway(*groups) -> InferenceResult:
    """
    One-way ANOVA: test whether all group means are equal.

    H₀: μ₁ = μ₂ = … = μ_k. Usually applied to three or more groups; with
    two groups it matches the pooled two-sample t-test (F = t²).

    Parameters
    ----------
    *groups : array-like
        Two or more groups of numeric data, passed as separate arguments.

    Returns
    -------
    InferenceResult
        ``statistic`` is F, with k − 1 and n − k degrees of freedom.
        ``summary()`` adds ``k`` and ``n``. No confidence interval.

    Examples
    --------
    >>> br = es.load_dataset("bridges")
    >>> groups = [g["load_capacity_kn"] for _, g in br.groupby("material")]
    >>> es.anova_oneway(*groups)
    """
    cleaned = [require_numeric_1d(g, f"group_{i+1}") for i, g in enumerate(groups)]
    f, p = stats.f_oneway(*cleaned)
    return InferenceResult("One-way ANOVA", f, p,
                           extra={"k": len(cleaned), "n": sum(len(g) for g in cleaned)})
