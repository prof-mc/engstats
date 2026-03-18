# Companion package `engstats`

**A companion Python package for freshman engineering statistics.**

The package `engstats` wraps `statsmodels`, `scipy`, `seaborn`, and `matplotlib` behind a clean, student-friendly API — sensible defaults, readable output, and descriptive error messages.

---

## Installation

```bash
pip install engstats
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

## Bundled Datasets

| Name       | Description                                         |
|------------|-----------------------------------------------------|
| `concrete` | Compressive strength vs. water-cement ratio + age   |
| `bridges`  | Load capacity vs. span length, by material          |
| `circuits` | Resistor tolerance measurements across batches      |

```python
es.load_dataset("list")   # print all available datasets
```

---

## Module Overview

```
engstats/
├── analysis/
│   ├── descriptive.py    # five_number_summary, summary_stats
│   ├── regression.py     # simple/multiple/polynomial regression
│   ├── inference.py      # t-tests, chi-square, ANOVA
│   ├── probability.py    # normal, binomial, Poisson probabilities
│   └── correlation.py    # Pearson, Spearman, correlation matrix
├── plots/
│   ├── theme.py          # auto-applied visual theme
│   ├── distributions.py  # histogram, KDE, boxplot, violin, ECDF
│   ├── regression.py     # scatter+line, residuals, Q-Q
│   ├── categorical.py    # bar, grouped bar
│   ├── probability.py    # normal curve, binomial PMF, CI
│   └── multivariate.py   # heatmap, pairplot
├── utils/
│   ├── validation.py     # student-friendly input checking
│   ├── formatting.py     # pretty_summary()
│   └── io.py             # load_dataset()
└── data/
    ├── concrete.csv
    ├── bridges.csv
    └── circuits.csv
```

---

## Running Tests

```bash
pytest tests/ -v
```

---

## Example Notebooks

See the `notebooks/` folder:

- `01_descriptive_stats.ipynb`
- `04_linear_regression.ipynb`
