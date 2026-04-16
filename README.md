# Companion package `engstats`

**A companion Python package for GEN_ENG 231 (IEMS 201) introductory engineering statistics class.**

The package `engstats` provides a companion package to handle making visualizations for an introductory engineering statistics class.

*A user manual is currently under construction.*

---

## Installation

```bash
pip install -U engstats
```

Or from source:

```bash
git clone https://github.com/prof-mc/engstats
cd engstats
pip install -e ".[dev]"
```

---

## Quick Start

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

# Probability
p = es.normal_prob(1.96)          # P(Z ≤ 1.96) ≈ 0.975
es.plot_normal_curve(mean=70, std=5, shade_above=80)
```

---

## Example Datasets

| Name       | Description                                         |
|------------|-----------------------------------------------------|
| `concrete` | Compressive strength vs. water-cement ratio + age   |
| `bridges`  | Load capacity vs. span length, by material          |
| `circuits` | Resistor tolerance measurements across batches      |

```python
es.load_dataset("list")   # print all available datasets
```

---

## Running Tests

```bash
pytest tests/ -v
```

---
