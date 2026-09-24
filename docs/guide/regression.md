# Regression

All regression functions fit ordinary least squares via `statsmodels` and return a
`RegressionResult`.

## Simple linear regression

\[
y = \beta_0 + \beta_1 x + \varepsilon
\]

```python
df = es.load_dataset("concrete")
model = es.simple_linear_regression(df, x="water_cement", y="strength_mpa")
model.summary()      # coefficients, SEs, CIs, p-values, ANOVA table
model.r_squared
model.coefficients
```

## Multiple linear regression

```python
model = es.multiple_linear_regression(df, x=["water_cement", "age_days"], y="strength_mpa")
model.summary()
```

## Polynomial regression

```python
model = es.polynomial_regression(df, x="water_cement", y="strength_mpa", degree=2)
```

## Diagnostics

```python
es.plot_scatter_regression(df, x="water_cement", y="strength_mpa")
es.plot_residuals(model)
es.plot_qq(model)
```

Residuals vs. fitted should show no pattern and constant spread. The Q-Q plot should
follow the reference line if the errors are approximately normal.
