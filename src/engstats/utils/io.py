"""
engstats.utils.io
==================
Dataset loader for bundled example datasets.
"""

import json
from importlib.resources import files
from pathlib import Path
import pandas as pd


_DATA_DIR = Path(__file__).parent.parent / "data"
_VALID_DATASETS: list[str] | None = None


def _get_valid_names() -> list[str]:
    global _VALID_DATASETS
    if _VALID_DATASETS is None:
        index_path = _DATA_DIR / "index.json"
        if index_path.exists():
            with open(index_path) as f:
                _VALID_DATASETS = list(json.load(f).keys())
        else:
            _VALID_DATASETS = [p.stem for p in _DATA_DIR.glob("*.csv")]
    return _VALID_DATASETS


def load_dataset(name: str) -> pd.DataFrame:
    """
    Load one of the bundled engineering datasets.

    Parameters
    ----------
    name : str
        Dataset name (without .csv extension).
        Call ``es.load_dataset('list')`` to print available datasets.

    Returns
    -------
    pd.DataFrame

    Examples
    --------
    >>> import engstats as es
    >>> df = es.load_dataset("concrete")
    >>> df.head()
    """
    if name == "list":
        names = _get_valid_names()
        print("Available datasets:")
        for n in names:
            print(f"  • {n}")
        return

    path = _DATA_DIR / f"{name}.csv"
    if not path.exists():
        valid = _get_valid_names()
        raise FileNotFoundError(
            f"Dataset '{name}' not found. "
            f"Available datasets: {valid}. "
            f"Tip: call es.load_dataset('list') to see all options."
        )
    return pd.read_csv(path)
