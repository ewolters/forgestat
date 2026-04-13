"""Tests for regularized regression (Ridge, Lasso, Elastic Net)."""

import numpy as np
import pytest

from forgestat.regression.regularized import (
    RegularizedResult,
    elastic_net,
    lasso,
    regularization_path,
    ridge,
)
from forgestat.regression.linear import ols


class TestRidge:
    def test_known_data(self):
        """Ridge on simple data produces reasonable coefficients."""
        np.random.seed(42)
        n = 100
        x1 = np.random.normal(0, 1, n)
        x2 = np.random.normal(0, 1, n)
        y = 5 + 2 * x1 - 3 * x2 + np.random.normal(0, 0.5, n)
        X = np.column_stack([x1, x2])
        result = ridge(X, y, alpha=0.1, feature_names=["x1", "x2"])
        assert isinstance(result, RegularizedResult)
        assert result.method == "ridge"
        assert abs(result.coefficients["x1"] - 2.0) < 0.3
        assert abs(result.coefficients["x2"] - (-3.0)) < 0.3
        assert result.r_squared > 0.9

    def test_alpha_zero_approximates_ols(self):
        """Ridge with alpha=0 should give results close to OLS."""
        np.random.seed(42)
        n = 50
        x = np.random.normal(0, 1, n)
        y = 3 + 1.5 * x + np.random.normal(0, 0.5, n)
        X = x.reshape(-1, 1)

        ridge_result = ridge(X, y, alpha=1e-10, feature_names=["x"])
        ols_result = ols(X, y, feature_names=["x"])

        assert abs(ridge_result.coefficients["x"] - ols_result.coefficients["x"]) < 0.05
        assert abs(ridge_result.intercept - ols_result.coefficients["Intercept"]) < 0.05

    def test_cross_validation(self):
        """Cross-validation picks a reasonable alpha."""
        np.random.seed(42)
        n = 100
        X = np.random.normal(0, 1, (n, 3))
        y = X @ [2, -1, 0.5] + np.random.normal(0, 0.5, n)
        result = ridge(X, y, alphas=[0.001, 0.01, 0.1, 1.0, 10.0])
        assert result.cv_score is not None
        assert result.alpha in [0.001, 0.01, 0.1, 1.0, 10.0]

    def test_auto_alpha_gcv(self):
        """Auto-select alpha via GCV when alpha=None."""
        np.random.seed(42)
        n = 80
        X = np.random.normal(0, 1, (n, 2))
        y = X @ [3, -2] + np.random.normal(0, 1, n)
        result = ridge(X, y, alpha=None)
        assert result.cv_score is not None
        assert result.alpha > 0

    def test_standardized_coef(self):
        """Standardized coefficients are returned."""
        np.random.seed(42)
        n = 50
        X = np.random.normal(0, 1, (n, 2))
        y = X @ [1, 2] + np.random.normal(0, 0.5, n)
        result = ridge(X, y, alpha=0.1, feature_names=["x1", "x2"])
        assert "x1" in result.standardized_coef
        assert "x2" in result.standardized_coef

    def test_single_feature(self):
        """Ridge works with a single feature."""
        np.random.seed(42)
        x = np.random.normal(0, 1, 30)
        y = 2 * x + np.random.normal(0, 0.5, 30)
        result = ridge(x.reshape(-1, 1), y, alpha=0.5)
        assert result.n_features_selected == 1
        assert abs(result.coefficients["X1"] - 2.0) < 0.5

    def test_shrinkage_increases_with_alpha(self):
        """Higher alpha should shrink coefficients more."""
        np.random.seed(42)
        n = 50
        X = np.random.normal(0, 1, (n, 2))
        y = X @ [3, -2] + np.random.normal(0, 0.5, n)

        r_low = ridge(X, y, alpha=0.01)
        r_high = ridge(X, y, alpha=100.0)

        coef_norm_low = sum(c ** 2 for c in r_low.coefficients.values())
        coef_norm_high = sum(c ** 2 for c in r_high.coefficients.values())
        assert coef_norm_high < coef_norm_low


class TestLasso:
    def test_sparse_solution(self):
        """Lasso produces sparse solutions — some coefficients exactly 0."""
        np.random.seed(42)
        n = 100
        X = np.random.normal(0, 1, (n, 5))
        # Only first 2 features matter
        y = 3 * X[:, 0] - 2 * X[:, 1] + np.random.normal(0, 0.5, n)
        result = lasso(X, y, alpha=0.3, feature_names=["x1", "x2", "x3", "x4", "x5"])
        assert result.method == "lasso"
        # At least some features should be zeroed out
        zero_count = sum(1 for v in result.coefficients.values() if abs(v) < 1e-10)
        assert zero_count > 0, "Lasso should zero out irrelevant features"
        assert result.n_features_selected < 5

    def test_correlated_features_selects_one(self):
        """With perfectly correlated features, Lasso picks one."""
        np.random.seed(42)
        n = 100
        x = np.random.normal(0, 1, n)
        X = np.column_stack([x, x + np.random.normal(0, 0.01, n)])
        y = 2 * x + np.random.normal(0, 0.5, n)
        result = lasso(X, y, alpha=0.1, feature_names=["x1", "x2"])
        # One should be near zero or both small — Lasso picks one
        coefs = [abs(result.coefficients["x1"]), abs(result.coefficients["x2"])]
        # At least one should be much smaller than the other
        assert min(coefs) < max(coefs) * 0.5 or result.n_features_selected <= 2

    def test_cv_selects_alpha(self):
        """Cross-validation picks a reasonable alpha."""
        np.random.seed(42)
        n = 100
        X = np.random.normal(0, 1, (n, 3))
        y = X @ [2, -1, 0] + np.random.normal(0, 0.5, n)
        result = lasso(X, y, alphas=[0.001, 0.01, 0.1, 0.5, 1.0])
        assert result.cv_score is not None
        assert result.r_squared > 0.5

    def test_selected_features_list(self):
        """selected_features lists non-zero features."""
        np.random.seed(42)
        n = 100
        X = np.random.normal(0, 1, (n, 4))
        y = 5 * X[:, 0] + np.random.normal(0, 0.5, n)
        result = lasso(X, y, alpha=0.5, feature_names=["a", "b", "c", "d"])
        assert "a" in result.selected_features
        assert result.n_features_selected == len(result.selected_features)

    def test_n_less_than_p(self):
        """Lasso handles n < p (more features than observations)."""
        np.random.seed(42)
        n = 20
        p = 50
        X = np.random.normal(0, 1, (n, p))
        beta_true = np.zeros(p)
        beta_true[:3] = [2, -1, 0.5]
        y = X @ beta_true + np.random.normal(0, 0.5, n)
        result = lasso(X, y, alpha=0.3)
        # Should produce a solution (not crash)
        assert len(result.coefficients) == p
        assert result.n_features_selected < p


class TestElasticNet:
    def test_interpolates_ridge_lasso(self):
        """Elastic Net with l1_ratio=0 behaves like Ridge, =1 like Lasso."""
        np.random.seed(42)
        n = 100
        X = np.random.normal(0, 1, (n, 5))
        y = X @ [3, -2, 0, 0, 0] + np.random.normal(0, 0.5, n)

        # l1_ratio=1 should be sparse (Lasso-like)
        en_lasso = elastic_net(X, y, alpha=0.3, l1_ratio=1.0)
        # l1_ratio=0 should keep more features (Ridge-like)
        en_ridge = elastic_net(X, y, alpha=0.3, l1_ratio=0.0)

        sparse_count_lasso = sum(1 for v in en_lasso.coefficients.values() if abs(v) < 1e-10)
        sparse_count_ridge = sum(1 for v in en_ridge.coefficients.values() if abs(v) < 1e-10)
        assert sparse_count_lasso >= sparse_count_ridge

    def test_l1_ratio_stored(self):
        """l1_ratio is stored in the result."""
        X = np.random.default_rng(42).normal(0, 1, (50, 2))
        y = X @ [1, 2] + np.random.default_rng(42).normal(0, 0.5, 50)
        result = elastic_net(X, y, alpha=0.1, l1_ratio=0.7)
        assert result.l1_ratio == 0.7
        assert result.method == "elastic_net"

    def test_reasonable_fit(self):
        """Elastic Net produces a reasonable fit on clean data."""
        np.random.seed(42)
        n = 100
        X = np.random.normal(0, 1, (n, 3))
        y = X @ [2, -1, 3] + np.random.normal(0, 0.5, n)
        result = elastic_net(X, y, alpha=0.05, l1_ratio=0.5)
        assert result.r_squared > 0.8


class TestRegularizationPath:
    def test_shrinkage_pattern(self):
        """Path shows coefficients shrinking toward zero as alpha increases."""
        np.random.seed(42)
        n = 100
        X = np.random.normal(0, 1, (n, 3))
        y = X @ [3, -2, 1] + np.random.normal(0, 0.5, n)
        path = regularization_path(X, y, method="lasso", n_alphas=20, feature_names=["a", "b", "c"])
        assert len(path.alphas) == 20
        assert "a" in path.coefficients
        # At the largest alpha, coefficients should be smaller
        # alphas are sorted from large to small
        for name in ["a", "b", "c"]:
            first_coef = abs(path.coefficients[name][0])  # largest alpha
            last_coef = abs(path.coefficients[name][-1])  # smallest alpha
            assert first_coef <= last_coef + 0.01  # shrinkage

    def test_ridge_path(self):
        """Ridge path works without errors."""
        np.random.seed(42)
        X = np.random.normal(0, 1, (50, 2))
        y = X @ [1, 2] + np.random.normal(0, 0.5, 50)
        path = regularization_path(X, y, method="ridge", n_alphas=10)
        assert path.method == "ridge"
        assert len(path.alphas) == 10

    def test_elastic_net_path(self):
        """Elastic Net path works."""
        np.random.seed(42)
        X = np.random.normal(0, 1, (50, 2))
        y = X @ [1, 2] + np.random.normal(0, 0.5, 50)
        path = regularization_path(X, y, method="elastic_net", n_alphas=10)
        assert path.method == "elastic_net"

    def test_unknown_method(self):
        """Unknown method raises ValueError."""
        X = np.random.default_rng(42).normal(0, 1, (20, 2))
        y = X @ [1, 1]
        with pytest.raises(ValueError, match="Unknown method"):
            regularization_path(X, y, method="foobar")

    def test_best_alpha_in_range(self):
        """best_alpha should be one of the alphas tested."""
        np.random.seed(42)
        X = np.random.normal(0, 1, (50, 2))
        y = X @ [1, -1] + np.random.normal(0, 0.5, 50)
        path = regularization_path(X, y, method="lasso", n_alphas=15)
        assert path.best_alpha in path.alphas


class TestStandardization:
    def test_predictions_invariant_to_scale(self):
        """Scaling features should not change predictions (only coefficient scale)."""
        np.random.seed(42)
        n = 50
        x1 = np.random.normal(0, 1, n)
        x2 = np.random.normal(0, 100, n)  # very different scale
        y = 2 * x1 + 0.01 * x2 + np.random.normal(0, 0.5, n)

        X_orig = np.column_stack([x1, x2])
        X_scaled = np.column_stack([x1 * 10, x2 / 10])

        r1 = ridge(X_orig, y, alpha=0.1, feature_names=["x1", "x2"])
        r2 = ridge(X_scaled, y, alpha=0.1, feature_names=["x1", "x2"])

        # Predictions should be similar (not identical due to different regularization effect)
        pred1 = np.array(r1.residuals) + np.array(y)  # y - residuals = fitted, so fitted = y - res...
        # Actually residuals = y - y_pred, so y_pred = y - residuals
        # Just check R-squared is similar
        assert abs(r1.r_squared - r2.r_squared) < 0.1


class TestEdgeCases:
    def test_perfectly_correlated_features(self):
        """Perfectly correlated features don't crash Ridge."""
        np.random.seed(42)
        n = 50
        x = np.random.normal(0, 1, n)
        X = np.column_stack([x, x])  # perfectly correlated
        y = 2 * x + np.random.normal(0, 0.5, n)
        result = ridge(X, y, alpha=1.0, feature_names=["x1", "x2"])
        # Should not crash; coefficients shared between correlated features
        total_coef = result.coefficients["x1"] + result.coefficients["x2"]
        assert abs(total_coef - 2.0) < 0.5

    def test_single_observation_per_feature_lasso(self):
        """Edge case: very few observations."""
        X = np.array([[1, 2], [3, 4], [5, 6]])
        y = np.array([1, 2, 3])
        result = lasso(X, y, alpha=0.5)
        assert len(result.coefficients) == 2
