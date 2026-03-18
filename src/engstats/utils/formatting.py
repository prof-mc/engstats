"""
engstats.utils.formatting
==========================
Helpers to produce clean, readable output from statsmodels and scipy objects.
"""

import pandas as pd


def pretty_summary(result) -> pd.DataFrame:
    """
    Convert an engstats RegressionResult or InferenceResult to a tidy DataFrame.

    Parameters
    ----------
    result : RegressionResult or InferenceResult

    Returns
    -------
    pd.DataFrame
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
