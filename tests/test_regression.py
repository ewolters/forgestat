"""Tests for regression module."""

import numpy as np

from forgestat.regression.linear import ols, polynomial
from forgestat.regression.nonlinear import curve_fit
from forgestat.regression.logistic import logistic_regression, poisson_regression
from forgestat.regression.stepwise import stepwise


class TestOLS:
    def test_perfect_fit(self):
        X = [[1], [2], [3], [4], [5]]
        y = [3, 5, 7, 9, 11]  # y = 1 + 2x
        result = ols(X, y, feature_names=["x"])
        assert abs(result.r_squared - 1.0) < 1e-10
        assert abs(result.coefficients["Intercept"] - 1.0) < 0.01
        assert abs(result.coefficients["x"] - 2.0) < 0.01

    def test_multiple_regression(self):
        np.random.seed(42)
        n = 100
        x1 = np.random.normal(0, 1, n)
        x2 = np.random.normal(0, 1, n)
        y = 5 + 2 * x1 - 3 * x2 + np.random.normal(0, 0.1, n)
        X = np.column_stack([x1, x2])
        result = ols(X, y, feature_names=["x1", "x2"])
        assert result.r_squared > 0.99
        assert abs(result.coefficients["x1"] - 2.0) < 0.1
        assert abs(result.coefficients["x2"] - (-3.0)) < 0.1

    def test_f_statistic(self):
        X = [[1], [2], [3], [4], [5]]
        y = [2, 4, 5, 4, 5]
        result = ols(X, y)
        assert result.f_statistic > 0
        assert 0 <= result.f_p_value <= 1

    def test_diagnostics(self):
        X = [[1], [2], [3], [4], [5]]
        y = [2, 4, 5, 4, 5]
        result = ols(X, y)
        assert len(result.residuals) == 5
        assert len(result.leverage) == 5
        assert len(result.cooks_distance) == 5
        assert 0 < result.durbin_watson < 4

    def test_p_values(self):
        np.random.seed(42)
        n = 50
        x = np.random.normal(0, 1, n)
        y = 2 * x + np.random.normal(0, 0.5, n)
        result = ols(x.reshape(-1, 1), y, feature_names=["x"])
        assert result.p_values["x"] < 0.001  # strong signal


class TestPolynomial:
    def test_quadratic(self):
        x = np.linspace(-3, 3, 50)
        y = 1 + 2 * x - 0.5 * x ** 2 + np.random.default_rng(42).normal(0, 0.1, 50)
        result = polynomial(x, y, degree=2)
        assert result.r_squared > 0.99
        assert "x" in result.coefficients
        assert "x^2" in result.coefficients


class TestNonlinear:
    def test_exponential(self):
        x = np.linspace(0, 5, 50)
        y = 2 * np.exp(0.5 * x) + np.random.default_rng(42).normal(0, 0.1, 50)
        result = curve_fit(x, y, model="exponential")
        assert result.converged
        assert result.r_squared > 0.99
        assert abs(result.parameters["a"] - 2.0) < 0.5

    def test_logistic(self):
        x = np.linspace(-5, 5, 100)
        y = 10 / (1 + np.exp(-1.5 * (x - 0))) + np.random.default_rng(42).normal(0, 0.1, 100)
        result = curve_fit(x, y, model="logistic", p0=[10, 1.5, 0])
        assert result.converged
        assert result.r_squared > 0.99

    def test_michaelis_menten(self):
        x = np.linspace(0.1, 10, 30)
        y = 100 * x / (5 + x) + np.random.default_rng(42).normal(0, 1, 30)
        result = curve_fit(x, y, model="michaelis_menten", p0=[100, 5])
        assert result.converged
        assert result.r_squared > 0.95

    def test_unknown_model(self):
        import pytest
        with pytest.raises(ValueError, match="Unknown model"):
            curve_fit([1, 2], [3, 4], model="foobar")


class TestLogistic:
    def test_overlapping_classes(self):
        np.random.seed(42)
        X = np.random.randn(200, 2)
        # Add noise so classes overlap — prevents perfect separation
        y = (X[:, 0] + X[:, 1] + np.random.randn(200) * 0.5 > 0).astype(int)
        result = logistic_regression(X, y, feature_names=["x1", "x2"])
        assert result.converged
        assert result.pseudo_r_squared > 0.2
        # Both coefficients should be positive (both contribute positively)
        assert result.coefficients["x1"] > 0
        assert result.coefficients["x2"] > 0

    def test_odds_ratios(self):
        np.random.seed(42)
        X = np.random.randn(200, 1)
        y = (X[:, 0] > 0).astype(int)
        result = logistic_regression(X, y, feature_names=["x"])
        assert result.odds_ratios["x"] > 1  # positive effect


class TestPoisson:
    def test_count_data(self):
        np.random.seed(42)
        X = np.random.randn(100, 1)
        y = np.random.poisson(np.exp(1 + 0.5 * X[:, 0]))
        result = poisson_regression(X, y, feature_names=["x"])
        assert result.n == 100
        assert "x" in result.coefficients
        assert result.irr["x"] > 1  # exp(positive coef)


class TestStepwise:
    def test_selects_correct_features(self):
        np.random.seed(42)
        n = 100
        x1 = np.random.normal(0, 1, n)
        x2 = np.random.normal(0, 1, n)
        x3 = np.random.normal(0, 1, n)  # noise
        y = 5 + 3 * x1 - 2 * x2 + np.random.normal(0, 0.5, n)
        X = np.column_stack([x1, x2, x3])
        result = stepwise(X, y, feature_names=["x1", "x2", "x3"], method="forward")
        assert "x1" in result.selected_features
        assert "x2" in result.selected_features
        assert result.final_model is not None
        assert result.final_model.r_squared > 0.9

    def test_backward(self):
        np.random.seed(42)
        n = 100
        x1 = np.random.normal(0, 1, n)
        x2 = np.random.normal(0, 1, n)
        y = 5 + 3 * x1 + np.random.normal(0, 0.5, n)
        X = np.column_stack([x1, x2])
        result = stepwise(X, y, feature_names=["x1", "x2"], method="backward")
        assert "x1" in result.selected_features
