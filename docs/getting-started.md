# Getting started

## Installation

=== "pip"

    ```bash
    pip install -U engstats
    ```

=== "uv"

    ```bash
    uv add engstats
    ```

=== "From source"

    ```bash
    git clone https://github.com/prof-mc/engstats
    cd engstats
    uv sync --extra dev
    ```

`engstats` requires Python 3.11 or newer.

## Importing

```python
import engstats as es
```

Importing the package applies the course plot theme globally (see
[`apply_theme`](reference/plots/theme.md)). Every public function is available at the
top level as `es.<name>`.

## A first analysis

```python
import engstats as es

# Load a bundled dataset
df = es.load_dataset("concrete")
df.head()

# Descriptive statistics
es.five_number_summary(df["strength_mpa"])
es.summary_stats(df["strength_mpa"])

# Simple linear regression
model = es.simple_linear_regression(df, x="water_cement", y="strength_mpa")
model.summary()

# Diagnostic plots
es.plot_scatter_regression(df, x="water_cement", y="strength_mpa")
es.plot_residuals(model)
es.plot_qq(model)

# Hypothesis testing
group_a = df[df["age_days"] <= 14]["strength_mpa"]
group_b = df[df["age_days"] >= 28]["strength_mpa"]
result = es.two_sample_ttest(group_a, group_b)
print(result)

# Probability (use scipy.stats directly)
from scipy import stats
p = stats.norm.cdf(1.96)          # P(Z <= 1.96) ~ 0.975
```
