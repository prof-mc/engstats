"""
engstats.analysis.regression
==============================
Linear and polynomial regression wrappers around statsmodels OLS.
Returns clean, student-friendly result objects.
"""

import numpy as np
import pandas as pd
import statsmodels.api as sm
from engstats.utils.validation import require_numeric_1d, require_dataframe


class RegressionResult:
    """
    Lightweight container for regression output.

    Attributes
    ----------
    coefficients : pd.Series
        Fitted coefficients (intercept first).
    r_squared : float
        Coefficient of determination (R²).
    adj_r_squared : float
        Adjusted R².
    p_values : pd.Series
        p-value for each coefficient.
    conf_int : pd.DataFrame
        95 % confidence intervals for each coefficient.
    residuals : np.ndarray
        Observed minus predicted values.
    fitted_values : np.ndarray
        Model predictions on the training data.
    model : statsmodels RegressionResults
        The underlying statsmodels object for advanced use.
    """

    def __init__(self, result, feature_names=None):
        self._result = result
        names = feature_names or result.model.exog_names
        self.coefficients = pd.Series(result.params, index=names)
        self.r_squared = result.rsquared
        self.adj_r_squared = result.rsquared_adj
        self.p_values = pd.Series(result.pvalues, index=names)
        self.conf_int = pd.DataFrame(result.conf_int()).set_axis(names)
        self.conf_int.columns = ["CI_lower", "CI_upper"]
        self.residuals = np.asarray(result.resid)
        self.fitted_values = np.asarray(result.fittedvalues)
        self.model = result

    def summary(self) -> pd.DataFrame:
        """Return a tidy coefficient table with estimates, p-values, and CIs."""
        df = pd.DataFrame(
            {
                "coefficient": self.coefficients,
                "p_value": self.p_values,
                "CI_lower": self.conf_int["CI_lower"],
                "CI_upper": self.conf_int["CI_upper"],
            }
        )
        df.index.name = "term"
        print(f"R² = {self.r_squared:.4f}   Adj. R² = {self.adj_r_squared:.4f}")
        return df

    def __repr__(self):
        return (
            f"RegressionResult(R²={self.r_squared:.4f}, "
            f"n={len(self.residuals)}, "
            f"terms={list(self.coefficients.index)})"
        )


def simple_linear_regression(data: pd.DataFrame, x: str, y: str) -> RegressionResult:
    """
    Fit a simple linear regression model: y = β₀ + β₁·x.

    Parameters
    ----------
    data : pd.DataFrame
        DataFrame containing both columns.
    x : str
        Name of the predictor column.
    y : str
        Name of the response column.

    Returns
    -------
    RegressionResult

    Examples
    --------
    >>> import engstats as es
    >>> df = es.load_dataset("concrete")
    >>> model = es.simple_linear_regression(df, x="water_cement", y="strength")
    >>> model.summary()
    """
    data = require_dataframe(data, [x, y])
    X = sm.add_constant(data[x].values)
    result = sm.OLS(data[y].values, X).fit()
    return RegressionResult(result, feature_names=["intercept", x])


def multiple_linear_regression(
    data: pd.DataFrame, predictors: list[str], response: str
) -> RegressionResult:
    """
    Fit a multiple linear regression model.

    Parameters
    ----------
    data : pd.DataFrame
        DataFrame containing all columns.
    predictors : list of str
        Names of predictor columns.
    response : str
        Name of the response column.

    Returns
    -------
    RegressionResult

    Examples
    --------
    >>> model = es.multiple_linear_regression(df, ["age", "water_cement"], "strength")
    """
    data = require_dataframe(data, predictors + [response])
    X = sm.add_constant(data[predictors].values)
    result = sm.OLS(data[response].values, X).fit()
    return RegressionResult(result, feature_names=["intercept"] + predictors)


def polynomial_regression(
    data: pd.DataFrame, x: str, y: str, degree: int = 2
) -> RegressionResult:
    """
    Fit a polynomial regression model of a given degree.

    Parameters
    ----------
    data : pd.DataFrame
    x : str
        Predictor column name.
    y : str
        Response column name.
    degree : int
        Polynomial degree (default 2).

    Returns
    -------
    RegressionResult

    Examples
    --------
    >>> model = es.polynomial_regression(df, x="temp", y="yield", degree=3)
    """
    data = require_dataframe(data, [x, y])
    x_vals = data[x].values
    feature_names = ["intercept"] + [f"{x}^{i}" for i in range(1, degree + 1)]
    X = np.column_stack([x_vals**i for i in range(1, degree + 1)])
    X = sm.add_constant(X)
    result = sm.OLS(data[y].values, X).fit()
    return RegressionResult(result, feature_names=feature_names)
