# Descriptive statistics

```python
import engstats as es
df = es.load_dataset("concrete")
```

## Five-number summary

The minimum, first quartile, median, third quartile, and maximum.

```python
es.five_number_summary(df["strength_mpa"], name="strength")
```

## Summary table

Mean, standard deviation, variance, and the quartiles in one table.

```python
es.summary_stats(df["strength_mpa"], name="strength")
```

## Seeing the distribution

```python
es.plot_histogram(df["strength_mpa"], kde=True, xlabel="Strength (MPa)")
es.plot_boxplot(values=df["strength_mpa"])
es.plot_ecdf(df["strength_mpa"], xlabel="Strength (MPa)")
```
