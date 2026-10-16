"""Tests for engstats.analysis.inference."""

import numpy as np
import pytest
from scipy import stats as scipy_stats
from engstats.analysis.inference import (
    one_sample_ttest,
    two_sample_ttest,
    paired_ttest,
    chi_square_test,
    anova_oneway,
)

rng = np.random.default_rng(1)
A = rng.normal(10, 2, 30)
B = rng.normal(12, 2, 30)


def test_one_sample_ttest_matches_scipy():
    result = one_sample_ttest(A, popmean=10)
    ref_t, ref_p = scipy_stats.ttest_1samp(A, 10)
    assert pytest.approx(result.statistic, rel=1e-5) == ref_t
    assert pytest.approx(result.p_value, rel=1e-5) == ref_p


def test_two_sample_ttest_detects_difference():
    result = two_sample_ttest(A, B)
    assert result.p_value < 0.05  # groups have different means


def test_two_sample_ttest_same_group():
    result = two_sample_ttest(A, A)
    assert result.p_value == pytest.approx(1.0, abs=1e-10)


def test_paired_ttest_length_mismatch():
    with pytest.raises(ValueError):
        paired_ttest([1, 2, 3], [4, 5])


def test_chi_square_uniform():
    obs = [25, 25, 25, 25]
    result = chi_square_test(obs)
    assert result.p_value == pytest.approx(1.0, abs=1e-10)


def test_anova_oneway_significant():
    g1 = rng.normal(5, 1, 20)
    g2 = rng.normal(10, 1, 20)
    g3 = rng.normal(15, 1, 20)
    result = anova_oneway(g1, g2, g3)
    assert result.p_value < 0.001


def test_chi_square_gof_reports_dof():
    result = chi_square_test([18, 22, 20, 40])
    assert result.summary()["dof"].iloc[0] == 3
    assert result.conf_int_low is None and result.conf_int_high is None


def test_chi_square_independence_reports_dof():
    result = chi_square_test([[10, 20, 30], [30, 40, 50]])
    assert result.summary()["dof"].iloc[0] == 2  # (2 - 1) * (3 - 1)
    assert result.conf_int_low is None


def test_anova_reports_k_and_n():
    result = anova_oneway([1, 2, 3], [2, 3, 4, 5], [5, 6, 7])
    row = result.summary().iloc[0]
    assert row["k"] == 3 and row["n"] == 10
    assert result.conf_int_low is None
