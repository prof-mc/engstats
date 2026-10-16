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
        Paired numeric data of equal length.

    Returns
    -------
    dict
        ``{"r": float, "p_value": float, "n": int}``. ``r`` and ``p_value``
        are rounded to 6 decimal places; ``p_value`` is two-sided for
        H₀: ρ = 0.

    Raises
    ------
    ValueError
        If ``x`` and ``y`` differ in length or contain non-numeric values.

    Examples
    --------
    >>> df = es.load_dataset("concrete")
    >>> es.pearson_r(df["water_cement"], df["strength_mpa"])
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

    Use instead of [`pearson_r`][engstats.analysis.correlation.pearson_r] when the relationship is monotonic but
    not linear, or when outliers dominate.

    Parameters
    ----------
    x, y : array-like
        Paired numeric data of equal length.

    Returns
    -------
    dict
        ``{"rho": float, "p_value": float, "n": int}``, rounded to 6 decimal
        places; ``p_value`` is two-sided for H₀: ρ = 0.

    Raises
    ------
    ValueError
        If ``x`` and ``y`` differ in length or contain non-numeric values.

    Examples
    --------
    >>> df = es.load_dataset("concrete")
    >>> es.spearman_rho(df["age_days"], df["strength_mpa"])
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

    Non-numeric columns are dropped before computing.

    Parameters
    ----------
    data : pd.DataFrame
        Must contain at least two numeric columns.
    method : {'pearson', 'spearman', 'kendall'}
        Correlation coefficient to compute. Default ``'pearson'``.

    Returns
    -------
    pd.DataFrame
        Symmetric correlation matrix, rounded to 4 decimal places.

    Raises
    ------
    ValueError
        If fewer than two numeric columns are present.

    Examples
    --------
    >>> es.correlation_matrix(df)
    """
    require_dataframe(data)
    numeric = data.select_dtypes(include="number")
    if numeric.shape[1] < 2:
        raise ValueError("DataFrame must contain at least 2 numeric columns.")
    return numeric.corr(method=method).round(4)
