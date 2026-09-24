# Plotting

Plot functions accept an optional `ax=` so they can be placed on a shared figure.

```python
import matplotlib.pyplot as plt
import engstats as es

df = es.load_dataset("bridges")
fig, axes = plt.subplots(1, 2, figsize=(10, 4))
es.plot_boxplot(df, x="material", y="load_capacity_kn", ax=axes[0])
es.plot_scatter_regression(df, x="span_m", y="load_capacity_kn", ax=axes[1])
```

| Topic          | Functions |
|----------------|-----------|
| Distributions  | `plot_histogram`, `plot_kde`, `plot_boxplot`, `plot_violin`, `plot_ecdf`, `plot_pareto`, `plot_side_by_side` |
| Regression     | `plot_scatter`, `plot_scatter_regression`, `plot_residuals`, `plot_qq`, `plot_runplot`, `plot_runplot_split` |
| Categorical    | `plot_bar`, `plot_grouped_bar`, `plot_stackeddot` |
| Multivariate   | `plot_correlation_heatmap`, `plot_pairplot` |

See the [API reference](../reference/index.md) for every argument.
