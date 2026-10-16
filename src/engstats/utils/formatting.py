"""
engstats.utils.formatting
==========================
Helpers to produce clean, readable output from statsmodels and scipy objects.
"""

import pandas as pd


def pretty_summary(result) -> pd.DataFrame:
    """
    Show the summary of an engstats result object.

    Dispatches to the object's ``.summary()`` method.

    Parameters
    ----------
    result : RegressionResult or InferenceResult
        Output of an engstats regression or inference function.

    Returns
    -------
    pd.DataFrame or None
        A one-row DataFrame for an ``InferenceResult``. For a
        ``RegressionResult`` the summary is printed and ``None`` is returned
        (see [`RegressionResult.summary`][engstats.analysis.regression.RegressionResult.summary]).

    Raises
    ------
    TypeError
        If ``result`` is neither result type.

    Examples
    --------
    >>> from engstats.utils import pretty_summary
    >>> pretty_summary(es.chi_square_test([18, 22, 20]))
    """
    # RegressionResult
    if hasattr(result, "coefficients"):
        return result.summary()

    # InferenceResult
    if hasattr(result, "statistic"):
        return result.summary()

    raise TypeError(
        f"pretty_summary() does not know how to format {type(result).__name__}. "
        f"Pass a RegressionResult or InferenceResult."
    )
