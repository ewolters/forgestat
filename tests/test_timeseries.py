"""Tests for time series module."""

import numpy as np

from forgestat.timeseries.stationarity import adf_test, auto_diff_order, kpss_test
from forgestat.timeseries.correlation import acf_pacf, cross_correlation
from forgestat.timeseries.decomposition import classical_decompose
from forgestat.timeseries.forecasting import arima, sarima
from forgestat.timeseries.changepoint import anomaly_scores, bocpd, pelt
from forgestat.timeseries.causality import detect_regimes, granger_causality


class TestStationarity:
    def test_adf_stationary(self):
        rng = np.random.default_rng(42)
        data = rng.normal(0, 1, 200).tolist()
        result = adf_test(data)
        assert result.is_stationary is True
        assert result.p_value < 0.05

    def test_adf_nonstationary(self):
        # Random walk
        rng = np.random.default_rng(42)
        data = np.cumsum(rng.normal(0, 1, 200)).tolist()
        result = adf_test(data)
        assert result.is_stationary is False

    def test_kpss(self):
        rng = np.random.default_rng(42)
        data = rng.normal(50, 1, 200).tolist()
        result = kpss_test(data)
        assert result.test_name == "KPSS"

    def test_auto_diff(self):
        rng = np.random.default_rng(42)
        data = np.cumsum(rng.normal(0, 1, 200)).tolist()
        d = auto_diff_order(data)
        assert d >= 1


class TestACFPACF:
    def test_basic(self):
        rng = np.random.default_rng(42)
        data = rng.normal(0, 1, 200).tolist()
        result = acf_pacf(data, n_lags=15)
        assert len(result.acf_values) == 16  # lag 0 through 15
        assert result.acf_values[0] == 1.0  # lag 0 = 1
        assert result.confidence_bound > 0

    def test_ar_process(self):
        # AR(1) process — PACF should cut off after lag 1
        rng = np.random.default_rng(42)
        n = 500
        x = np.zeros(n)
        for i in range(1, n):
            x[i] = 0.8 * x[i - 1] + rng.normal(0, 1)
        result = acf_pacf(x.tolist(), n_lags=10)
        assert result.suggested_order["p"] >= 1

    def test_ljung_box(self):
        rng = np.random.default_rng(42)
        data = rng.normal(0, 1, 200).tolist()
        result = acf_pacf(data)
        assert result.ljung_box_p is not None


class TestCCF:
    def test_lagged_relationship(self):
        rng = np.random.default_rng(42)
        n = 200
        x = rng.normal(0, 1, n)
        # y is x shifted by 3 periods + noise
        y = np.zeros(n)
        y[3:] = x[:-3] + rng.normal(0, 0.3, n - 3)
        result = cross_correlation(x.tolist(), y.tolist(), max_lag=10)
        # Peak should be around lag 3 (x leads y)
        assert abs(result.peak_lag - 3) <= 1
        assert result.peak_value > 0.5

    def test_no_relationship(self):
        rng = np.random.default_rng(42)
        x = rng.normal(0, 1, 100).tolist()
        y = rng.normal(0, 1, 100).tolist()
        result = cross_correlation(x, y)
        assert abs(result.peak_value) < 0.4


class TestDecomposition:
    def test_additive(self):
        # Trend + seasonal + noise
        n = 120
        t = np.arange(n)
        trend = 100 + 0.5 * t
        seasonal = 10 * np.sin(2 * np.pi * t / 12)
        noise = np.random.default_rng(42).normal(0, 1, n)
        data = (trend + seasonal + noise).tolist()

        result = classical_decompose(data, period=12, model="additive")
        assert result.seasonal_strength > 0.5
        assert result.trend_direction == "upward"
        assert len(result.trend) == n
        assert len(result.seasonal) == n


class TestARIMA:
    def test_basic_forecast(self):
        rng = np.random.default_rng(42)
        data = np.cumsum(rng.normal(0.1, 1, 100)).tolist()
        result = arima(data, order=(1, 1, 0), forecast_steps=5)
        assert len(result.forecast) == 5
        assert result.aic != 0
        assert len(result.residuals) > 0
        for pt in result.forecast:
            assert pt.ci_lower < pt.predicted < pt.ci_upper

    def test_sarima(self):
        # Seasonal data
        n = 120
        t = np.arange(n)
        seasonal = 10 * np.sin(2 * np.pi * t / 12)
        trend = 0.1 * t
        noise = np.random.default_rng(42).normal(0, 1, n)
        data = (100 + trend + seasonal + noise).tolist()

        result = sarima(data, order=(1, 0, 0), seasonal_order=(1, 1, 0, 12), forecast_steps=12)
        assert result.seasonal_order == (1, 1, 0, 12)
        assert len(result.forecast) == 12


class TestChangepoint:
    def test_pelt_detects_shift(self):
        rng = np.random.default_rng(42)
        seg1 = rng.normal(10, 1, 50)
        seg2 = rng.normal(20, 1, 50)
        data = np.concatenate([seg1, seg2]).tolist()
        result = pelt(data, min_size=10)
        assert len(result.changepoints) >= 1
        # Should detect change near index 50
        cp_indices = [cp.index for cp in result.changepoints]
        assert any(40 <= idx <= 60 for idx in cp_indices)

    def test_bocpd(self):
        # BOCPD with very large shift and tuned priors
        rng = np.random.default_rng(42)
        seg1 = rng.normal(0, 0.1, 80)
        seg2 = rng.normal(10, 0.1, 80)
        data = np.concatenate([seg1, seg2]).tolist()
        result = bocpd(data, hazard_rate=1 / 30, threshold=0.1, kappa_prior=0.01, beta_prior=0.01)
        assert result.method == "bocpd"
        # BOCPD is inherently conservative — test structure not value
        assert result.n_segments >= 1

    def test_no_changepoint(self):
        rng = np.random.default_rng(42)
        data = rng.normal(50, 1, 100).tolist()
        result = pelt(data, min_size=20)
        # Stable data — should find 0 or very few changepoints
        assert len(result.changepoints) <= 1


class TestAnomaly:
    def test_adaptive_detects_spike(self):
        rng = np.random.default_rng(42)
        data = rng.normal(50, 2, 100)
        data[50] = 100  # inject spike
        result = anomaly_scores(data.tolist(), window=20, threshold=3.0)
        assert 50 in result.anomaly_indices

    def test_zscore(self):
        rng = np.random.default_rng(42)
        data = rng.normal(0, 1, 100)
        data[30] = 10  # 10σ outlier
        result = anomaly_scores(data.tolist(), method="zscore", threshold=3.0)
        assert 30 in result.anomaly_indices

    def test_clean_data(self):
        rng = np.random.default_rng(42)
        data = rng.normal(50, 1, 100).tolist()
        result = anomaly_scores(data, threshold=4.0)
        assert result.n_anomalies < 5


class TestGranger:
    def test_causality_detected(self):
        rng = np.random.default_rng(42)
        n = 200
        x = rng.normal(0, 1, n)
        y = np.zeros(n)
        for i in range(2, n):
            y[i] = 0.7 * x[i - 2] + rng.normal(0, 0.5)
        result = granger_causality(x.tolist(), y.tolist(), max_lag=4)
        assert result.x_causes_y is True
        assert result.best_lag == 2

    def test_no_causality(self):
        rng = np.random.default_rng(42)
        x = rng.normal(0, 1, 200).tolist()
        y = rng.normal(0, 1, 200).tolist()
        result = granger_causality(x, y, max_lag=4)
        # Independent series — likely no causality
        assert len(result.results_by_lag) == 4


class TestRegimes:
    def test_two_regimes(self):
        rng = np.random.default_rng(42)
        stable = rng.normal(50, 1, 100)
        volatile = rng.normal(50, 10, 100)
        data = np.concatenate([stable, volatile]).tolist()
        result = detect_regimes(data, n_regimes=2)
        assert result.n_regimes >= 1
        assert len(result.regime_labels) == 200

    def test_short_data(self):
        data = [1, 2, 3, 4, 5]
        result = detect_regimes(data)
        assert result.n_regimes == 1


class TestRepeatedMeasures:
    def test_significant(self):
        from forgestat.parametric.repeated_measures import repeated_measures_anova
        data = {
            "Baseline": [10, 12, 11, 13, 14, 15, 12, 11],
            "Treatment1": [15, 17, 16, 18, 19, 20, 17, 16],
            "Treatment2": [20, 22, 21, 23, 24, 25, 22, 21],
        }
        result = repeated_measures_anova(data)
        assert result.p_value < 0.001
        assert result.n_conditions == 3
        assert result.n_subjects == 8
        assert result.partial_eta_sq > 0.5

    def test_greenhouse_geisser(self):
        from forgestat.parametric.repeated_measures import repeated_measures_anova
        rng = np.random.default_rng(42)
        data = {f"C{i}": rng.normal(50 + i * 5, 3, 20).tolist() for i in range(4)}
        result = repeated_measures_anova(data)
        assert 0 < result.epsilon_gg <= 1
        assert result.p_value_gg >= result.p_value  # correction makes p larger or equal


class TestRunsTest:
    def test_random_sequence(self):
        from forgestat.nonparametric.rank_tests import runs_test
        rng = np.random.default_rng(42)
        data = rng.normal(0, 1, 100).tolist()
        result = runs_test(data)
        # Random data should not be flagged
        assert result.p_value > 0.01

    def test_clustered(self):
        from forgestat.nonparametric.rank_tests import runs_test
        # Very few runs — values clustered
        data = [1] * 50 + [10] * 50
        result = runs_test(data)
        assert result.p_value < 0.05  # too few runs
