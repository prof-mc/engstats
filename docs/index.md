# engstats

**A companion Python package for GEN_ENG 231, introductory engineering statistics.**

`engstats` wraps `statsmodels`, `scipy`, `seaborn`, and `matplotlib` behind a small set of
student-friendly functions. Each analysis returns a tidy result object, and each plot
returns (or draws on) a Matplotlib `Axes` with a consistent course theme.

```python
import engstats as es

df = es.load_dataset("concrete")
model = es.simple_linear_regression(df, x="water_cement", y="strength_mpa")
model.summary()
es.plot_scatter_regression(df, x="water_cement", y="strength_mpa")
```

<div class="grid cards" markdown>

-   **Getting started**

    Install the package and run your first analysis.

    [:octicons-arrow-right-24: Getting started](getting-started.md)

-   **User guide**

    Worked examples organised by course topic.

    [:octicons-arrow-right-24: User guide](guide/index.md)

-   **API reference**

    Every public function, generated from the docstrings.

    [:octicons-arrow-right-24: API reference](reference/index.md)

</div>

!!! note "Draft"
    This manual is under construction. The API reference is complete; the user guide
    pages are being written alongside the course.
