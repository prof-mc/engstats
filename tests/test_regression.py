"""Tests for engstats.analysis.regression."""

import numpy as np
import pandas as pd
import pytest
from engstats.analysis.regression import (
    simple_linear_regression,
    multiple_linear_regression,
    polynomial_regression,
)

rng = np.random.default_rng(0)
N = 50
x = rng.uniform(0, 10, N)
y = 3.0 + 2.5 * x + rng.normal(0, 1, N)
df = pd.DataFrame({"x": x, "y": y})


def test_simple_linear_regression_coefficients():
    model = simple_linear_regression(df, x="x", y="y")
    assert pytest.approx(model.coefficients["intercept"], abs=0.5) == 3.0
    assert pytest.approx(model.coefficients["x"], abs=0.3) == 2.5


def test_simple_linear_regression_r2():
    model = simple_linear_regression(df, x="x", y="y")
    assert model.r_squared > 0.95


def test_residuals_length():
    model = simple_linear_regression(df, x="x", y="y")
    assert len(model.residuals) == N


def test_multiple_regression():
    z = rng.uniform(0, 5, N)
    df2 = df.copy()
    df2["z"] = z
    df2["y"] = 1.0 + 2.0 * df2["x"] + 1.5 * z + rng.normal(0, 0.5, N)
    model = multiple_linear_regression(df2, predictors=["x", "z"], response="y")
    assert "x" in model.coefficients.index
    assert "z" in model.coefficients.index
    assert model.r_squared > 0.9


def test_polynomial_regression():
    model = polynomial_regression(df, x="x", y="y", degree=2)
    assert "x^2" in model.coefficients.index


def test_missing_column_raises():
    with pytest.raises(KeyError):
        simple_linear_regression(df, x="nonexistent", y="y")
