"""Tests for engstats.analysis.descriptive."""

import numpy as np
import pytest
from engstats.analysis.descriptive import five_number_summary, summary_stats


DATA = [2, 4, 6, 8, 10, 12, 14]


def test_five_number_summary_shape():
    result = five_number_summary(DATA)
    assert result.shape == (1, 5)
    assert list(result.columns) == ["Min", "Q1", "Median", "Q3", "Max"]


def test_five_number_summary_values():
    result = five_number_summary(DATA)
    assert result["Min"].iloc[0] == 2
    assert result["Max"].iloc[0] == 14
    assert result["Median"].iloc[0] == pytest.approx(8.0)


def test_summary_stats_keys():
    result = summary_stats(DATA)
    for col in ["n", "mean", "std", "variance", "min", "median", "max", "IQR", "skewness", "kurtosis"]:
        assert col in result.columns


def test_summary_stats_n():
    result = summary_stats(DATA)
    assert result["n"].iloc[0] == len(DATA)


def test_invalid_input_strings():
    with pytest.raises(ValueError):
        five_number_summary(["a", "b", "c"])


def test_empty_input():
    with pytest.raises(ValueError):
        five_number_summary([])
