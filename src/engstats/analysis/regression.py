"""
engstats.analysis.regression
==============================
Linear and polynomial regression wrappers around statsmodels OLS.
Returns clean, student-friendly result objects.
"""

import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.formula.api import ols
from engstats.utils.validation import require_numeric_1d, require_dataframe


class RegressionResult:
    """
    Lightweight container for regression output.

    Returned by every regression function in this module.

    Attributes
    ----------
    n : int
        Number of observations used in the fit.
    coefficients : pd.Series
        Fitted coefficients, intercept first. Index labels are ``Intercept``
        plus column names for linear models, and ``intercept``, ``x^1``,
        ``x^2``, ... for [`polynomial_regression`][engstats.analysis.regression.polynomial_regression].
    se_coefficients : pd.Series
        Standard error of each coefficient.
    r_squared : float
        Coefficient of determination (R²).
    adj_r_squared : float
        Adjusted R².
    p_values : pd.Series
        p-value for each coefficient.
    conf_int : pd.DataFrame
        95 % confidence intervals for each coefficient, with columns
        ``95% CI_lower`` and ``95% CI_upper``.
    residuals : np.ndarray
        Observed minus predicted values.
    fitted_values : np.ndarray
        Model predictions on the training data.
    """

    def __init__(self, result, feature_names=None):
        self._result = result
        names = feature_names or result.model.exog_names
        self.n = len(result.resid)
        # np.asarray: relabel by position, not by statsmodels' own term names
        self.coefficients = pd.Series(np.asarray(result.params), index=names)
        self.se_coefficients = pd.Series(np.asarray(result.bse), index=names)
        self.r_squared = result.rsquared
        self.adj_r_squared = result.rsquared_adj
        self.p_values = pd.Series(np.asarray(result.pvalues), index=names)
        self.conf_int = pd.DataFrame(result.conf_int()).set_axis(names)
        self.conf_int.columns = ["95% CI_lower", "95% CI_upper"]
        self.residuals = np.asarray(result.resid)
        self.fitted_values = np.asarray(result.fittedvalues)

    def summary(self):
        """
        Print the fitted model summary.

        Prints n, R² and adjusted R², a coefficient table (``coefficient``,
        ``SE coef``, 95 % CI bounds, ``p_value``, and ``crit_value``, the
        t-statistic for each coefficient), and the ANOVA table.

        Returns
        -------
        None
            Output is printed, not returned. Use the attributes
            (``coefficients``, ``p_values``, ...) to work with the numbers.
        """
        # print(model.)
        pd.set_option("display.precision", 4)
        df = pd.DataFrame(
            {
                "coefficient": self.coefficients,
                "SE coef": self.se_coefficients,
                "CI[0.025]": self.conf_int["95% CI_lower"],
                "CI[0.975]": self.conf_int["95% CI_upper"],
                "p_value": self.p_values,
                "crit_value": np.asarray(self._result.tvalues),
            }
        )
        df.index.name = "term"
        print(f"Number of observations (n): {self.n}")
        print(f"R² = {self.r_squared:.4f}   Adj. R² = {self.adj_r_squared:.4f}")

        print(df)
        print("----------------------------------------")
        print(pd.DataFrame(sm.stats.anova_lm(self._result)))

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

    Raises
    ------
    KeyError
        If ``x`` or ``y`` is not a column of ``data``.

    Examples
    --------
    >>> import engstats as es
    >>> df = es.load_dataset("concrete")
    >>> model = es.simple_linear_regression(df, x="water_cement", y="strength_mpa")
    >>> model.summary()
    """
    data = require_dataframe(data, [x, y])

    formula = f'{y} ~ {x}'
    result = ols(formula=formula, data=data).fit()
    #
    # if with_constant:
    #     X = sm.add_constant(data[x].values)
    # else:
    #     X = data[x].values
    # result = sm.OLS(data[y].values, X).fit()
    return RegressionResult(result)


def multiple_linear_regression(
    data: pd.DataFrame, x: list[str], y: str
) -> RegressionResult:
    """
    Fit a multiple linear regression model: y = β₀ + β₁·x₁ + … + β_p·x_p.

    Parameters
    ----------
    data : pd.DataFrame
        DataFrame containing all columns.
    x : list of str
        Names of predictor columns.
    y : str
        Name of the response column.

    Returns
    -------
    RegressionResult

    Examples
    --------
    >>> df = es.load_dataset("concrete")
    >>> model = es.multiple_linear_regression(df, x=["age_days", "water_cement"], y="strength_mpa")
    >>> model.summary()
    """
    data = require_dataframe(data, x + [y])

    formula = f'{y} ~ {" + ".join(x)}'
    result = ols(formula=formula, data=data).fit()

    # X = sm.add_constant(data[predictors].values)
    # result = sm.OLS(data[response].values, X).fit()
    return RegressionResult(result)


def polynomial_regression(
    data: pd.DataFrame, x: str, y: str, degree: int = 2
) -> RegressionResult:
    """
    Fit a polynomial regression model: y = β₀ + β₁·x + β₂·x² + … + β_d·x^d.

    Parameters
    ----------
    data : pd.DataFrame
        DataFrame containing both columns.
    x : str
        Predictor column name.
    y : str
        Response column name.
    degree : int
        Polynomial degree, 1 or more (default 2).

    Returns
    -------
    RegressionResult
        Coefficients are labelled ``intercept``, ``<x>^1``, ..., ``<x>^degree``.

    Raises
    ------
    KeyError
        If ``x`` or ``y`` is not a column of ``data``.
    ValueError
        If ``degree`` is not a positive integer.

    Examples
    --------
    >>> df = es.load_dataset("concrete")
    >>> model = es.polynomial_regression(df, x="water_cement", y="strength_mpa", degree=2)
    >>> model.coefficients
    """
    data = require_dataframe(data, [x, y])
    if not (isinstance(degree, int) and degree >= 1):
        raise ValueError(f"'degree' must be a positive integer, got {degree!r}.")
    # Fit through a formula (not a bare design matrix) so summary() can build
    # the ANOVA table, which needs the formula's term structure.
    terms = [x] + [f"I({x} ** {i})" for i in range(2, degree + 1)]
    result = ols(formula=f"{y} ~ {' + '.join(terms)}", data=data).fit()
    feature_names = ["intercept"] + [f"{x}^{i}" for i in range(1, degree + 1)]
    return RegressionResult(result, feature_names=feature_names)
