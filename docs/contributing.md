# Contributing

## Development setup
The package uses [uv](https://docs.astral.sh/uv/) to manage the Python package and dependencies.

```bash
git clone https://github.com/prof-mc/engstats
cd engstats
uv sync --extra dev
uv run pytest tests/ -v
```

## Building the docs

The docs use [MkDocs](https://www.mkdocs.org/) with the
[Material](https://squidfunk.github.io/mkdocs-material/) theme and
[mkdocstrings](https://mkdocstrings.github.io/) for the API reference.

```bash
uv sync --group docs          # install docs toolchain
uv run mkdocs serve           # live preview at http://127.0.0.1:8000
uv run mkdocs build --strict  # full build into site/, warnings are errors
```

API pages are generated from docstrings, so write them in
[NumPy style](https://numpydoc.readthedocs.io/en/latest/format.html):

```python
def pearson_r(x, y):
    """
    Compute the Pearson correlation coefficient and p-value.

    Parameters
    ----------
    x, y : array-like
        Paired numeric samples of equal length.

    Returns
    -------
    InferenceResult
    """
```

New modules need a one-line stub in `docs/reference/` and an entry under `nav:` in
`mkdocs.yml`.
