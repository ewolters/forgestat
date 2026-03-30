"""Tests for core modules — types, assumptions, effect_size, distributions."""

import numpy as np

from forgestat.core.assumptions import check_equal_variance, check_normality, check_outliers
from forgestat.core.effect_size import (
    classify_effect,
    cohens_d_one_sample,
    cohens_d_paired,
    cohens_d_two_sample,
    cramers_v,
    eta_squared,
    omega_squared,
    partial_eta_squared,
    rank_biserial,
)
from forgestat.core.distributions import box_cox, fit_best, fit_distribution


class TestAssumptions:
    def test_normality_normal_data(self):
        rng = np.random.default_rng(42)
        data = rng.normal(50, 5, 100).tolist()
        result = check_normality(data)
        assert result.name == "normality"
        assert result.passed is True

    def test_normality_skewed_data(self):
        rng = np.random.default_rng(42)
        data = rng.exponential(1, 200).tolist()
        result = check_normality(data)
        assert result.passed is False
        assert "normality" in result.suggestion.lower() or "non-parametric" in result.suggestion.lower()

    def test_normality_insufficient_data(self):
        result = check_normality([1.0, 2.0])
        assert result.passed is True
        assert "too few" in result.detail

    def test_normality_anderson(self):
        rng = np.random.default_rng(42)
        data = rng.normal(0, 1, 100).tolist()
        result = check_normality(data, method="anderson")
        assert result.test_name == "Anderson-Darling"

    def test_equal_variance_similar(self):
        rng = np.random.default_rng(42)
        g1 = rng.normal(50, 5, 50).tolist()
        g2 = rng.normal(55, 5, 50).tolist()
        result = check_equal_variance(g1, g2)
        assert result.passed is True

    def test_equal_variance_different(self):
        rng = np.random.default_rng(42)
        g1 = rng.normal(50, 2, 100).tolist()
        g2 = rng.normal(50, 20, 100).tolist()
        result = check_equal_variance(g1, g2)
        assert result.passed is False

    def test_outliers_clean_data(self):
        data = list(range(1, 101))
        result = check_outliers(data)
        assert result.passed is True

    def test_outliers_with_extremes(self):
        data = list(range(1, 51)) + [500, 600, 700, 800, 900, 1000]
        result = check_outliers(data)
        assert "outlier" in result.detail.lower()


class TestEffectSize:
    def test_cohens_d_one_sample(self):
        data = [10, 12, 11, 13, 14, 12, 11, 10]
        d = cohens_d_one_sample(data, mu=10)
        assert d > 0  # mean > 10, so d should be positive

    def test_cohens_d_two_sample(self):
        x1 = [10, 12, 11, 13, 14]
        x2 = [20, 22, 21, 23, 24]
        d = cohens_d_two_sample(x1, x2)
        assert abs(d) > 2  # very large effect

    def test_cohens_d_paired(self):
        x1 = [10, 12, 11, 13, 14]
        x2 = [11, 13, 12, 14, 15]
        d = cohens_d_paired(x1, x2)
        # Constant difference of -1 → sd=0 → returns -inf
        assert d != 0
        # Variable differences
        x3 = [10, 12, 11, 13, 14]
        x4 = [11, 14, 12, 15, 16]
        d2 = cohens_d_paired(x3, x4)
        assert d2 < 0  # x3 < x4

    def test_classify_effect(self):
        assert classify_effect(0.1, "cohens_d") == "negligible"
        assert classify_effect(0.3, "cohens_d") == "small"
        assert classify_effect(0.6, "cohens_d") == "medium"
        assert classify_effect(1.0, "cohens_d") == "large"

    def test_eta_squared(self):
        assert eta_squared(100, 400) == 0.25

    def test_omega_squared(self):
        w2 = omega_squared(100, 400, 2, 10)
        assert 0 < w2 < 0.25  # less biased, should be smaller

    def test_partial_eta_squared(self):
        assert partial_eta_squared(50, 50) == 0.5

    def test_cramers_v(self):
        v = cramers_v(10.0, 100, 2)
        assert 0 < v < 1

    def test_rank_biserial(self):
        r = rank_biserial(10, 5, 5)
        assert -1 <= r <= 1


class TestDistributions:
    def test_fit_normal(self):
        rng = np.random.default_rng(42)
        data = rng.normal(50, 5, 200).tolist()
        result = fit_distribution(data, "normal")
        assert result.name == "normal"
        assert result.ks_p_value > 0.01

    def test_fit_best(self):
        rng = np.random.default_rng(42)
        data = rng.normal(50, 5, 200).tolist()
        result = fit_best(data)
        assert result.best.name in ("normal", "lognormal", "gamma", "weibull")
        assert len(result.all_fits) >= 3
        assert result.data_summary["n"] == 200

    def test_box_cox(self):
        rng = np.random.default_rng(42)
        data = rng.exponential(10, 100).tolist()
        transformed, lmbda = box_cox(data)
        assert len(transformed) == 100
        assert isinstance(lmbda, float)
