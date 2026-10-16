# API reference

Generated from the NumPy-style docstrings in `src/engstats`. Functions marked **es**
are importable from the top level, e.g. `es.summary_stats`; the rest are imported from
their module.

| Module | Contents | Top level |
|--------|----------|:---------:|
| [`analysis.descriptive`](analysis/descriptive.md) | `five_number_summary`, `summary_stats` | es |
| [`analysis.inference`](analysis/inference.md) | `one_sample_ttest`, `two_sample_ttest`, `paired_ttest`, `chi_square_test`, `anova_oneway` | es |
| [`analysis.regression`](analysis/regression.md) | `simple_linear_regression`, `multiple_linear_regression`, `polynomial_regression` | es |
| [`analysis.correlation`](analysis/correlation.md) | `pearson_r`, `spearman_rho`, `correlation_matrix` | es |
| [`plots.distributions`](plots/distributions.md) | `plot_histogram`, `plot_kde`, `plot_boxplot`, `plot_violin`, `plot_ecdf`, `plot_probability`, `plot_pareto`, `plot_side_by_side` | es |
| [`plots.regression`](plots/regression.md) | `plot_scatter`, `plot_scatter_regression`, `plot_residuals`, `plot_qq`, `plot_runplot`, `plot_runplot_split` | es |
| [`plots.categorical`](plots/categorical.md) | `plot_bar`, `plot_grouped_bar`, `plot_stackeddot` | es |
| [`plots.multivariate`](plots/multivariate.md) | `plot_correlation_heatmap`, `plot_pairplot` | es |
| [`plots.inference`](plots/inference.md) | `plot_confidence_interval` | es |
| [`plots.theme`](plots/theme.md) | `apply_theme` | es |
| [`utils.io`](utils/io.md) | `load_dataset` | es |
| [`utils.validation`](utils/validation.md) | `require_numeric_1d`, `require_dataframe` | |
| [`utils.formatting`](utils/formatting.md) | `pretty_summary` | |
