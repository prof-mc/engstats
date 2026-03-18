"""
engstats.analysis.descriptive
==============================
Descriptive statistics helpers with student-friendly output.
"""

import numpy as np
import pandas as pd
from scipy import stats
from engstats.utils.validation import require_numeric_1d


def five_number_summary(data, name: str = "x") -> pd.DataFrame:
    """
    Return the five-number summary of a 1-D dataset.

    Parameters
    ----------
    data : array-like
        Numeric data (list, numpy array, or pandas Series).
    name : str
        Label used in the output table.

    Returns
    -------
    pd.DataFrame
        Single-row DataFrame with Min, Q1, Median, Q3, Max.

    Examples
    --------
    >>> import engstats as es
    >>> es.five_number_summary([2, 4, 6, 8, 10])
    """
    arr = require_numeric_1d(data, "data")
    q1, median, q3 = np.percentile(arr, [25, 50, 75])
    return pd.DataFrame(
        {
            "Min": [arr.min()],
            "Q1": [q1],
            "Median": [median],
            "Q3": [q3],
            "Max": [arr.max()],
        },
        index=[name],
    )


def summary_stats(data, name: str = "x") -> pd.DataFrame:
    """
    Return a comprehensive descriptive statistics table.

    Includes n, mean, std, variance, min, Q1, median, Q3, max,
    IQR, skewness, and kurtosis.

    Parameters
    ----------
    data : array-like
        Numeric data.
    name : str
        Label used in the output table.

    Returns
    -------
    pd.DataFrame
        Single-row DataFrame with all descriptive statistics.

    Examples
    --------
    >>> import engstats as es
    >>> es.summary_stats([3, 7, 7, 19])
    """
    arr = require_numeric_1d(data, "data")
    q1, median, q3 = np.percentile(arr, [25, 50, 75])
    return pd.DataFrame(
        {
            "n": [len(arr)],
            "mean": [arr.mean()],
            "std": [arr.std(ddof=1)],
            "variance": [arr.var(ddof=1)],
            "min": [arr.min()],
            "Q1": [q1],
            "median": [median],
            "Q3": [q3],
            "max": [arr.max()],
            "IQR": [q3 - q1],
            "skewness": [stats.skew(arr)],
            "kurtosis": [stats.kurtosis(arr)],
        },
        index=[name],
    )
