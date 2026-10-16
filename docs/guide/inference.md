# Hypothesis testing

Every test returns an `InferenceResult` with `statistic`, `p_value`, and (for t-tests)
a confidence interval in `conf_int_low` / `conf_int_high`. Call `.summary()` for a
one-row DataFrame.

| Test | The confidence interval covers |
|------|--------------------------------|
| `one_sample_ttest` | the population mean \(\mu\) |
| `two_sample_ttest` | the difference \(\mu_a - \mu_b\) |
| `paired_ttest` | the mean difference \(\mu_d\) = mean(before − after) |

## One-sample t-test

\(H_0: \mu = \mu_0\)

```python
df = es.load_dataset("circuits")
result = es.one_sample_ttest(df["measured_ohm"], popmean=100)
es.plot_confidence_interval(result)
```

`plot_confidence_interval` works with any t-test result and marks the null-hypothesis
value with a dashed line, so you can see whether the interval excludes it.

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
