"""Tests for Bayesian inference module."""

import numpy as np

from forgestat.bayesian.tests import (
    bayes_factor_shadow,
    bayesian_correlation,
    bayesian_proportion,
    bayesian_ttest_one_sample,
    bayesian_ttest_two_sample,
)


class TestBayesianTTest:
    def test_one_sample_strong_evidence(self):
        data = [10, 12, 11, 13, 14, 15, 12, 11, 10, 13]
        result = bayesian_ttest_one_sample(data, mu=0)
        assert result.bf10 > 10  # strong evidence against H0
        assert result.bf_label in ("strong", "very_strong", "extreme")

    def test_one_sample_null_true(self):
        rng = np.random.default_rng(42)
        data = rng.normal(50, 5, 30).tolist()
        result = bayesian_ttest_one_sample(data, mu=50)
        # BF should be near 1 or favor H0
        assert result.bf10 < 10

    def test_one_sample_credible_interval(self):
        data = [10, 12, 11, 13, 14]
        result = bayesian_ttest_one_sample(data, mu=0)
        assert result.credible_interval is not None
        lo, hi = result.credible_interval
        assert lo < hi
        assert lo < result.posterior_mean < hi

    def test_one_sample_rope(self):
        data = [0.1, -0.1, 0.2, -0.2, 0.05, -0.05, 0.15, -0.15]
        result = bayesian_ttest_one_sample(data, mu=0, rope=(-0.5, 0.5))
        assert result.p_rope is not None
        assert result.p_rope > 0.5  # most of posterior inside ROPE

    def test_two_sample_different(self):
        x1 = [10, 12, 11, 13, 14, 15, 12, 11]
        x2 = [20, 22, 21, 23, 24, 25, 22, 21]
        result = bayesian_ttest_two_sample(x1, x2)
        assert result.bf10 > 10

    def test_two_sample_similar(self):
        rng = np.random.default_rng(42)
        x1 = rng.normal(50, 5, 30).tolist()
        x2 = rng.normal(50, 5, 30).tolist()
        result = bayesian_ttest_two_sample(x1, x2)
        assert result.bf10 < 10


class TestBayesianProportion:
    def test_strong_evidence(self):
        result = bayesian_proportion(successes=80, n=100)
        assert result.posterior_mean > 0.7
        assert result.credible_interval[0] > 0.5

    def test_uniform_prior(self):
        result = bayesian_proportion(successes=50, n=100)
        assert abs(result.posterior_mean - 0.5) < 0.05

    def test_credible_interval(self):
        result = bayesian_proportion(successes=30, n=100)
        lo, hi = result.credible_interval
        assert lo < result.posterior_mean < hi
        assert 0 <= lo and hi <= 1


class TestBayesianCorrelation:
    def test_strong_correlation(self):
        x = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        y = [2, 4, 6, 8, 10, 12, 14, 16, 18, 20]
        result = bayesian_correlation(x, y)
        assert result.bf10 > 10
        assert result.posterior_mean > 0.9

    def test_no_correlation(self):
        rng = np.random.default_rng(42)
        x = rng.normal(0, 1, 50).tolist()
        y = rng.normal(0, 1, 50).tolist()
        result = bayesian_correlation(x, y)
        assert abs(result.posterior_mean) < 0.4


class TestBayesShadow:
    def test_large_t(self):
        result = bayes_factor_shadow(t_statistic=5.0, n=30)
        assert result["bf10"] > 10
        assert "alternative" in result["interpretation"]

    def test_small_t(self):
        result = bayes_factor_shadow(t_statistic=0.5, n=30)
        assert result["bf10"] < 10

    def test_zero_t(self):
        result = bayes_factor_shadow(t_statistic=0.0, n=30)
        assert result["bf10"] == 1.0
