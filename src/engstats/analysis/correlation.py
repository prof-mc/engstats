"""
engstats.analysis.correlation
==============================
Correlation coefficients and matrices.
"""

import numpy as np
import pandas as pd
from scipy import stats
from engstats.utils.validation import require_numeric_1d, require_dataframe


def pearson_r(x, y) -> dict:
    """
    Compute the Pearson correlation coefficient and p-value.

    Parameters
    ----------
    x, y : array-like
        Paired numeric data (equal length).

    Returns
    -------
    dict with keys: r, p_value, n

    Examples
    --------
    >>> es.pearson_r(df["x"], df["y"])
    """
    x = require_numeric_1d(x, "x")
    y = require_numeric_1d(y, "y")
    if len(x) != len(y):
        raise ValueError(f"x and y must be the same length (got {len(x)} and {len(y)}).")
    r, p = stats.pearsonr(x, y)
    return {"r": round(float(r), 6), "p_value": round(float(p), 6), "n": len(x)}


def spearman_rho(x, y) -> dict:
    """
    Compute the Spearman rank correlation coefficient and p-value.

    Parameters
    ----------
    x, y : array-like

    Returns
    -------
    dict with keys: rho, p_value, n
    """
    x = require_numeric_1d(x, "x")
    y = require_numeric_1d(y, "y")
    if len(x) != len(y):
        raise ValueError(f"x and y must be the same length (got {len(x)} and {len(y)}).")
    rho, p = stats.spearmanr(x, y)
    return {"rho": round(float(rho), 6), "p_value": round(float(p), 6), "n": len(x)}


def correlation_matrix(data: pd.DataFrame, method: str = "pearson") -> pd.DataFrame:
    """
    Compute a pairwise correlation matrix for all numeric columns.

    Parameters
    ----------
    data : pd.DataFrame
    method : {'pearson', 'spearman', 'kendall'}

    Returns
    -------
    pd.DataFrame
        Symmetric correlation matrix.

    Examples
    --------
    >>> es.correlation_matrix(df)
    """
    require_dataframe(data)
    numeric = data.select_dtypes(include="number")
    if numeric.shape[1] < 2:
        raise ValueError("DataFrame must contain at least 2 numeric columns.")
    return numeric.corr(method=method).round(4)
