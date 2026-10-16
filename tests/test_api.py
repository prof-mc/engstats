"""Tests for the top-level ``engstats`` namespace."""

import inspect

import engstats as es


def test_all_lists_every_top_level_function():
    public = {
        name for name, obj in vars(es).items()
        if inspect.isfunction(obj)
        and obj.__module__.startswith("engstats.")
        and not name.startswith("_")
        and name != "apply_theme"  # applied on import, not part of the student API
    }
    assert public - set(es.__all__) == set()


def test_all_names_exist():
    assert [n for n in es.__all__ if not hasattr(es, n)] == []
