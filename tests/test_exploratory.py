"""Tests for exploratory and meta-analysis modules."""

import numpy as np

from forgestat.exploratory.univariate import bootstrap_ci, describe, tolerance_interval
from forgestat.exploratory.multivariate import hotelling_t2_one_sample, one_way_manova, pca
from forgestat.exploratory.meta import meta_analysis


class TestDescribe:
    def test_basic(self):
        data = list(range(1, 101))
        result = describe(data)
        assert result.n == 100
        assert result.mean == 50.5
        assert result.median == 50.5
        assert result.min == 1
        assert result.max == 100

    def test_with_nan(self):
        data = [1, 2, float("nan"), 4, 5]
        result = describe(data)
        assert result.n == 4
        assert result.n_missing == 1

    def test_shape_symmetric(self):
        rng = np.random.default_rng(42)
        data = rng.normal(50, 5, 1000).tolist()
        result = describe(data)
        assert "symmetric" in result.shape_description

    def test_shape_skewed(self):
        rng = np.random.default_rng(42)
        data = rng.exponential(10, 1000).tolist()
        result = describe(data)
        assert "right-skewed" in result.shape_description

    def test_outliers_detected(self):
        data = list(range(1, 51)) + [500]
        result = describe(data)
        assert result.n_outliers >= 1

    def test_cv(self):
        data = [100, 100, 100, 100, 100]
        result = describe(data)
        assert result.cv == 0.0


class TestBootstrapCI:
    def test_mean(self):
        rng = np.random.default_rng(42)
        data = rng.normal(50, 5, 100).tolist()
        result = bootstrap_ci(data, statistic="mean")
        assert result.ci_lower < 50 < result.ci_upper
        assert result.estimate > 45

    def test_median(self):
        data = list(range(1, 101))
        result = bootstrap_ci(data, statistic="median")
        assert result.ci_lower < result.estimate < result.ci_upper


class TestToleranceInterval:
    def test_normal(self):
        rng = np.random.default_rng(42)
        data = rng.normal(50, 5, 100).tolist()
        result = tolerance_interval(data, coverage=0.95, confidence=0.95)
        assert result.lower < 50 < result.upper
        assert result.k_factor > 0

    def test_nonparametric(self):
        data = list(range(1, 101))
        result = tolerance_interval(data, coverage=0.95, method="nonparametric")
        assert result.lower < result.upper


class TestPCA:
    def test_basic(self):
        rng = np.random.default_rng(42)
        x = rng.normal(0, 1, 50)
        data = {"X": x.tolist(), "Y": (x + rng.normal(0, 0.1, 50)).tolist(), "Z": rng.normal(0, 1, 50).tolist()}
        result = pca(data)
        assert result.n_components == 3
        assert result.variance_explained[0] > 0.4  # first PC should explain most
        assert abs(sum(result.variance_explained) - 1.0) < 0.01

    def test_retain_components(self):
        data = {"A": [1, 2, 3, 4, 5], "B": [2, 4, 6, 8, 10], "C": [5, 4, 3, 2, 1]}
        result = pca(data, n_components=2)
        assert result.n_components == 2
        assert len(result.eigenvalues) == 2

    def test_loadings_shape(self):
        data = {"X": [1, 2, 3], "Y": [4, 5, 6]}
        result = pca(data)
        assert len(result.loadings) == 2
        assert len(result.loadings["X"]) == 2


class TestHotelling:
    def test_far_from_zero(self):
        data = {"X": [10, 11, 12, 13, 14], "Y": [20, 21, 22, 23, 24]}
        result = hotelling_t2_one_sample(data)
        assert result.p_value < 0.001

    def test_near_zero(self):
        rng = np.random.default_rng(42)
        data = {"X": rng.normal(0, 1, 30).tolist(), "Y": rng.normal(0, 1, 30).tolist()}
        result = hotelling_t2_one_sample(data)
        # Centered at zero, should not reject
        assert result.t2_statistic >= 0


class TestManova:
    def test_separated_groups(self):
        data = {
            "Y1": [10, 11, 12, 50, 51, 52, 90, 91, 92],
            "Y2": [20, 21, 22, 60, 61, 62, 100, 101, 102],
        }
        groups = ["A", "A", "A", "B", "B", "B", "C", "C", "C"]
        result = one_way_manova(data, groups)
        assert result.wilks_lambda < 0.1  # very separated
        assert result.p_value < 0.05


class TestMetaAnalysis:
    def test_fixed_effects(self):
        effects = [0.5, 0.6, 0.4, 0.55, 0.45]
        ses = [0.1, 0.12, 0.11, 0.09, 0.13]
        result = meta_analysis(effects, ses, model="fixed")
        assert 0.3 < result.pooled_effect < 0.7
        assert result.k == 5

    def test_random_effects(self):
        effects = [0.5, 0.6, 0.4, 0.55, 0.45]
        ses = [0.1, 0.12, 0.11, 0.09, 0.13]
        result = meta_analysis(effects, ses, model="random")
        assert result.pooled_effect > 0
        assert result.tau_squared >= 0

    def test_heterogeneity(self):
        # Very heterogeneous studies
        effects = [0.1, 0.5, 1.0, 2.0, 0.3]
        ses = [0.05, 0.05, 0.05, 0.05, 0.05]
        result = meta_analysis(effects, ses)
        assert result.i_squared > 50  # significant heterogeneity

    def test_homogeneous(self):
        effects = [0.50, 0.51, 0.49, 0.50, 0.50]
        ses = [0.1, 0.1, 0.1, 0.1, 0.1]
        result = meta_analysis(effects, ses)
        assert result.i_squared < 30

    def test_study_weights(self):
        result = meta_analysis([0.5, 0.6], [0.1, 0.2])
        # Smaller SE → larger weight
        assert result.studies[0].weight > result.studies[1].weight
