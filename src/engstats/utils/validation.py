"""
engstats.utils.validation
==========================
Input validation helpers with student-friendly error messages.
"""

import numpy as np
import pandas as pd


def require_numeric_1d(data, name: str = "data") -> np.ndarray:
    """
    Convert `data` to a 1-D float numpy array.

    Raises descriptive ValueErrors for common student mistakes:
    - passing a list of strings
    - passing a 2-D array when a 1-D array is expected
    - passing None or an empty array
    """
    if data is None:
        raise ValueError(f"'{name}' cannot be None — did you forget to pass your data?")

    # Accept pandas Series
    if isinstance(data, pd.Series):
        data = data.values

    try:
        arr = np.asarray(data, dtype=float)
    except (ValueError, TypeError):
        raise ValueError(
            f"'{name}' must contain only numbers. "
            f"Got: {type(data).__name__}. "
            f"Check that your list or column does not contain strings or NaN labels."
        )

    if arr.ndim != 1:
        raise ValueError(
            f"'{name}' must be 1-dimensional (a single column or list), "
            f"but got shape {arr.shape}. "
            f"Tip: pass a single column like df['{name}'], not the whole DataFrame."
        )

    if len(arr) == 0:
        raise ValueError(f"'{name}' is empty — make sure your data loaded correctly.")

    if np.all(np.isnan(arr)):
        raise ValueError(f"'{name}' contains only NaN values.")

    return arr


def require_dataframe(data, columns: list[str] | None = None) -> pd.DataFrame:
    """
    Confirm `data` is a DataFrame, optionally checking that `columns` exist.
    """
    if not isinstance(data, pd.DataFrame):
        raise TypeError(
            f"Expected a pandas DataFrame, got {type(data).__name__}. "
            f"Tip: load your data with pd.read_csv() or es.load_dataset()."
        )
    if columns:
        missing = [c for c in columns if c not in data.columns]
        if missing:
            raise KeyError(
                f"Column(s) not found in DataFrame: {missing}. "
                f"Available columns: {list(data.columns)}."
            )
    return data
