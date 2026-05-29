"""Tests for Bayesian inference module."""

import numpy as np

from forgestat.bayesian.tests import (
    bayes_factor_shadow,
    bayesian_ab,
    bayesian_anova,
    bayesian_correlation,
    bayesian_proportion,
    bayesian_regression,
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


class TestBayesianAnova:
    def test_clear_group_differences(self):
        rng = np.random.default_rng(1)
        groups = {
            "A": rng.normal(10, 1, 20).tolist(),
            "B": rng.normal(15, 1, 20).tolist(),
            "C": rng.normal(20, 1, 20).tolist(),
        }
        result = bayesian_anova(groups)
        assert result.bf10 > 10
        assert result.bf_label in ("strong", "very_strong", "extreme")
        assert 0.0 <= result.posterior_mean <= 1.0  # eta-squared
        assert result.extra["df_between"] == 2

    def test_no_group_differences(self):
        rng = np.random.default_rng(2)
        groups = {k: rng.normal(10, 2, 25).tolist() for k in ("A", "B", "C")}
        result = bayesian_anova(groups)
        assert result.bf10 < 3  # no/anecdotal evidence

    def test_accepts_list_of_lists(self):
        result = bayesian_anova([[1, 2, 3, 2, 1], [8, 9, 10, 9, 8]])
        assert result.bf10 > 1
        assert result.extra["k"] == 2


class TestBayesianRegression:
    def test_strong_predictor(self):
        rng = np.random.default_rng(3)
        x = rng.normal(0, 1, 60)
        y = 3 * x + rng.normal(0, 0.5, 60)
        result = bayesian_regression(y.tolist(), x.reshape(-1, 1).tolist())
        assert result.bf10 > 10
        assert result.posterior_mean > 0.8  # R^2
        assert len(result.extra["coefficients"]) == 1

    def test_no_relationship(self):
        rng = np.random.default_rng(4)
        x = rng.normal(0, 1, 60)
        y = rng.normal(0, 1, 60)
        result = bayesian_regression(y.tolist(), x.reshape(-1, 1).tolist())
        assert result.bf10 < 3

    def test_multiple_predictors(self):
        rng = np.random.default_rng(5)
        x1 = rng.normal(0, 1, 80)
        x2 = rng.normal(0, 1, 80)
        y = 2 * x1 - 1.5 * x2 + rng.normal(0, 0.5, 80)
        X = np.column_stack([x1, x2]).tolist()
        result = bayesian_regression(y.tolist(), X)
        assert result.bf10 > 10
        assert len(result.extra["coefficients"]) == 2


class TestBayesianAB:
    def test_variant_clearly_better(self):
        rng = np.random.default_rng(6)
        a = rng.normal(10, 2, 50).tolist()
        b = rng.normal(13, 2, 50).tolist()
        result = bayesian_ab(a, b)
        assert result.extra["prob_b_better"] > 0.9
        assert result.posterior_mean > 0  # B - A uplift

    def test_identical_arms_are_symmetric(self):
        rng = np.random.default_rng(7)
        a = rng.normal(10, 2, 50).tolist()
        b = list(reversed(a))  # same values → no real difference
        result = bayesian_ab(a, b)
        assert result.extra["prob_b_better"] == 0.5
        assert result.posterior_mean == 0.0

    def test_works_on_binary_conversions(self):
        a = [0, 1, 0, 0, 1, 0, 0, 0, 1, 0] * 5  # ~30%
        b = [1, 1, 0, 1, 1, 0, 1, 1, 0, 1] * 5  # ~70%
        result = bayesian_ab(a, b)
        assert result.extra["prob_b_better"] > 0.95
