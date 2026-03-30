"""Tests for reliability module."""

import numpy as np

from forgestat.reliability.distributions import (
    exponential_fit,
    hazard_function,
    lognormal_fit,
    reliability_function,
    weibull_fit,
)
from forgestat.reliability.survival import kaplan_meier, log_rank_test


class TestWeibull:
    def test_fit(self):
        rng = np.random.default_rng(42)
        data = rng.weibull(2.5, 100) * 1000  # shape=2.5, scale≈1000
        result = weibull_fit(data.tolist())
        assert result.shape > 1.5  # should be near 2.5
        assert result.scale > 0
        assert result.b10_life > 0
        assert result.mean_life > 0
        assert result.failure_mode == "wear_out"

    def test_infant_mortality(self):
        rng = np.random.default_rng(42)
        data = rng.weibull(0.5, 50) * 500  # shape < 1
        result = weibull_fit(data.tolist())
        assert result.shape < 1
        assert result.failure_mode == "infant_mortality"

    def test_ks_goodness(self):
        rng = np.random.default_rng(42)
        data = rng.weibull(3, 200) * 1000
        result = weibull_fit(data.tolist())
        assert result.ks_p_value > 0.01  # good fit


class TestReliabilityFunction:
    def test_decreasing(self):
        times = [100, 500, 1000, 2000, 5000]
        r = reliability_function(2.0, 1000, times)
        vals = [r[t] for t in times]
        assert all(vals[i] >= vals[i + 1] for i in range(len(vals) - 1))
        assert vals[0] > 0.5
        assert vals[-1] < 0.5

    def test_exponential(self):
        r = reliability_function(1.0, 1000, [1000], distribution="exponential")
        assert abs(r[1000] - np.exp(-1)) < 0.01  # R(MTTF) ≈ 0.368


class TestHazardFunction:
    def test_weibull_increasing(self):
        times = [100, 500, 1000, 2000]
        h = hazard_function(2.5, 1000, times)
        vals = [h[t] for t in times]
        assert all(vals[i] <= vals[i + 1] for i in range(len(vals) - 1))

    def test_exponential_constant(self):
        times = [100, 500, 1000, 2000]
        h = hazard_function(1.0, 1000, times, distribution="exponential")
        vals = [h[t] for t in times]
        assert all(abs(v - vals[0]) < 1e-10 for v in vals)


class TestExponentialFit:
    def test_basic(self):
        rng = np.random.default_rng(42)
        data = rng.exponential(500, 100).tolist()
        result = exponential_fit(data)
        assert result.distribution == "exponential"
        assert abs(result.mttf - 500) < 100  # should be near 500


class TestLognormalFit:
    def test_basic(self):
        rng = np.random.default_rng(42)
        data = rng.lognormal(6, 0.5, 100).tolist()
        result = lognormal_fit(data)
        assert result.distribution == "lognormal"
        assert result.mttf > 0
        assert result.b10 > 0


class TestKaplanMeier:
    def test_no_censoring(self):
        times = [10, 20, 30, 40, 50, 60, 70]
        result = kaplan_meier(times)
        assert result.n_total == 7
        assert result.n_events == 7
        assert result.n_censored == 0
        assert result.curve[-1].survival < 0.2

    def test_with_censoring(self):
        times = [10, 20, 30, 40, 50, 60, 70]
        events = [True, False, True, True, False, True, False]
        result = kaplan_meier(times, events)
        assert result.n_events == 4
        assert result.n_censored == 3
        # Last obs is censored, so survival should stay above 0
        assert result.curve[-1].survival > 0

    def test_median_survival(self):
        times = list(range(1, 21))
        result = kaplan_meier(times)
        assert result.median_survival is not None
        assert result.median_survival == 10  # median of 1-20

    def test_confidence_intervals(self):
        times = list(range(1, 51))
        result = kaplan_meier(times, ci_level=0.95)
        for pt in result.curve[1:]:
            assert pt.ci_lower <= pt.survival <= pt.ci_upper


class TestLogRank:
    def test_different_groups(self):
        # Group 1 fails early, Group 2 fails late
        t1 = [5, 10, 15, 20, 25, 30, 35, 40]
        e1 = [True] * 8
        t2 = [50, 60, 70, 80, 90, 100, 110, 120]
        e2 = [True] * 8
        result = log_rank_test(t1, e1, t2, e2)
        assert result.p_value < 0.05  # significantly different

    def test_similar_groups(self):
        rng = np.random.default_rng(42)
        t1 = rng.exponential(50, 30).tolist()
        e1 = [True] * 30
        t2 = rng.exponential(50, 30).tolist()
        e2 = [True] * 30
        result = log_rank_test(t1, e1, t2, e2)
        # Same distribution, likely not significant
        assert result.chi_square >= 0
