# Hypothesis testing

Every test returns an `InferenceResult` with `statistic`, `p_value`, and (for t-tests)
a confidence interval. Call `.summary()` for a one-row DataFrame.

## One-sample t-test

\(H_0: \mu = \mu_0\)

```python
df = es.load_dataset("circuits")
es.one_sample_ttest(df["measured_ohm"], popmean=100)
```

## Two-sample t-test

Welch's test (unequal variances) is the default. Pass `equal_var=True` for the pooled test.

```python
a = df.loc[df["batch"] == "A", "measured_ohm"]
b = df.loc[df["batch"] == "B", "measured_ohm"]
result = es.two_sample_ttest(a, b)
result.summary()
```

## Paired t-test

```python
es.paired_ttest(df["nominal_ohm"], df["measured_ohm"])
```

## Chi-square test

```python
es.chi_square_test(observed=[18, 22, 20], expected=[20, 20, 20])
```

## One-way ANOVA

```python
br = es.load_dataset("bridges")
groups = [g["load_capacity_kn"] for _, g in br.groupby("material")]
es.anova_oneway(*groups)
```
