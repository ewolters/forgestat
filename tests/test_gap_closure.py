"""Tests for gap closure — P1, P2, P3 analyses."""

import numpy as np

# P1
from forgestat.parametric.split_plot import split_plot_anova
from forgestat.quality.anom import anom
from forgestat.regression.robust import robust_regression

# P2
from forgestat.posthoc.comparisons import scheffe
from forgestat.exploratory.multi_vari import multi_vari
from forgestat.parametric.variance import f_test, variance_test
from forgestat.power.sample_size import power_equivalence, power_z_test, sample_size_tolerance

# P3
from forgestat.reliability.cox import cox_ph
from forgestat.regression.glm import glm, ordinal_logistic, orthogonal_regression
from forgestat.regression.best_subsets import best_subsets
from forgestat.msa.kappa import fleiss_kappa, krippendorff_alpha


class TestSplitPlot:
    def test_basic(self):
        rng = np.random.default_rng(42)
        data = {
            "response": [],
            "operator": [],
            "setting": [],
        }
        for op in ["A", "B"]:
            for setting in ["low", "med", "high"]:
                for _ in range(4):
                    base = 10 * (1 if op == "A" else 2) + 5 * ["low", "med", "high"].index(setting)
                    data["response"].append(base + rng.normal(0, 1))
                    data["operator"].append(op)
                    data["setting"].append(setting)

        result = split_plot_anova(data, "response", "operator", "setting")
        assert len(result.sources) == 4  # WP, SP, WP×SP, Residual
        assert result.n_whole_plots == 2
        assert result.n_sub_plots == 3


class TestANOM:
    def test_detects_outlier_group(self):
        g1 = [10, 11, 10, 11, 10]
        g2 = [10, 11, 10, 11, 10]
        g3 = [50, 51, 50, 51, 50]  # way outside
        result = anom(g1, g2, g3, labels=["A", "B", "C"])
        assert result.any_significant
        outlier = [g for g in result.groups if g.exceeds_upper]
        assert len(outlier) >= 1
        assert outlier[0].name == "C"

    def test_all_similar(self):
        rng = np.random.default_rng(42)
        groups = [rng.normal(50, 1, 20).tolist() for _ in range(3)]
        result = anom(*groups)
        assert not result.any_significant


class TestRobustRegression:
    def test_with_outliers(self):
        rng = np.random.default_rng(42)
        n = 50
        x = rng.normal(0, 1, n)
        y = 2 * x + 1 + rng.normal(0, 0.5, n)
        # Inject outliers
        y[0] = 100
        y[1] = -100

        result = robust_regression(x.reshape(-1, 1), y, feature_names=["x"])
        assert result.n_downweighted >= 2
        assert abs(result.coefficients["x"] - 2) < 1  # should be close to true slope

    def test_bisquare(self):
        x = np.array([[1], [2], [3], [4], [5]], dtype=float)
        y = [2, 4, 6, 8, 100]  # one outlier
        result = robust_regression(x, y, method="bisquare")
        assert result.method == "bisquare"


class TestScheffe:
    def test_conservative(self):
        g1 = [10, 12, 11, 13, 14]
        g2 = [20, 22, 21, 23, 24]
        g3 = [30, 32, 31, 33, 34]
        result = scheffe(g1, g2, g3)
        assert result.test_name == "scheffe"
        assert len(result.comparisons) == 3


class TestMultiVari:
    def test_identifies_dominant(self):
        rng = np.random.default_rng(42)
        data = {
            "response": [],
            "machine": [],
            "shift": [],
        }
        for m in ["M1", "M2", "M3"]:
            for s in ["Day", "Night"]:
                base = {"M1": 10, "M2": 50, "M3": 90}[m]  # machine dominates
                for _ in range(10):
                    data["response"].append(base + rng.normal(0, 2))
                    data["machine"].append(m)
                    data["shift"].append(s)

        result = multi_vari(data, "response", ["machine", "shift"])
        assert result.dominant_source == "machine"


class TestVarianceTests:
    def test_f_test(self):
        rng = np.random.default_rng(42)
        x1 = rng.normal(0, 1, 50).tolist()
        x2 = rng.normal(0, 5, 50).tolist()
        result = f_test(x1, x2)
        assert result.p_value < 0.05  # very different variances

    def test_levene(self):
        rng = np.random.default_rng(42)
        g1 = rng.normal(0, 1, 50).tolist()
        g2 = rng.normal(0, 5, 50).tolist()
        result = variance_test(g1, g2, method="levene")
        assert result.p_value < 0.05


class TestPowerWrappers:
    def test_power_z(self):
        result = power_z_test(effect_size=0.5, n=64)
        assert result.power > 0.5

    def test_power_equivalence(self):
        result = power_equivalence(effect_size=0.1, margin=0.5, n=50)
        assert result.power > 0.3

    def test_sample_size_tolerance(self):
        n = sample_size_tolerance(coverage=0.95, confidence=0.95)
        assert n > 30


class TestCoxPH:
    def test_basic(self):
        rng = np.random.default_rng(42)
        n = 100
        x = rng.normal(0, 1, n)
        times = rng.exponential(10, n) * np.exp(-0.5 * x)  # x increases hazard
        events = [True] * n
        result = cox_ph(times.tolist(), events, {"treatment": x.tolist()})
        assert result.n == n
        assert "treatment" in result.coefficients


class TestGLM:
    def test_gaussian(self):
        X = [[1], [2], [3], [4], [5]]
        y = [2, 4, 6, 8, 10]
        result = glm(X, y, feature_names=["x"], family="gaussian")
        assert result.family == "gaussian"
        assert abs(result.coefficients.get("x", 0) - 2) < 0.1

    def test_ordinal(self):
        rng = np.random.default_rng(42)
        X = rng.normal(0, 1, (60, 1))
        y = [0] * 20 + [1] * 20 + [2] * 20
        result = ordinal_logistic(X, y, feature_names=["x"])
        assert result.n == 60
        assert len(result.categories) == 3

    def test_orthogonal(self):
        x = [1, 2, 3, 4, 5]
        y = [2.1, 3.9, 6.1, 7.9, 10.1]
        result = orthogonal_regression(x, y)
        assert abs(result.slope - 2) < 0.5


class TestBestSubsets:
    def test_finds_true_predictors(self):
        rng = np.random.default_rng(42)
        n = 100
        x1 = rng.normal(0, 1, n)
        x2 = rng.normal(0, 1, n)
        x3 = rng.normal(0, 1, n)  # noise
        y = 5 + 3 * x1 - 2 * x2 + rng.normal(0, 0.5, n)
        X = np.column_stack([x1, x2, x3])
        result = best_subsets(X, y, feature_names=["x1", "x2", "noise"])
        assert result.best_bic is not None
        # Best model should include x1 and x2
        assert "x1" in result.best_bic.features
        assert "x2" in result.best_bic.features


class TestKrippendorff:
    def test_perfect_agreement(self):
        ratings = [[1, 2, 3, 4], [1, 2, 3, 4], [1, 2, 3, 4]]
        result = krippendorff_alpha(ratings, level="nominal")
        assert result.value > 0.9

    def test_no_agreement(self):
        ratings = [[1, 2, 3], [3, 1, 2], [2, 3, 1]]
        result = krippendorff_alpha(ratings, level="nominal")
        assert result.value < 0.3


class TestFleiss:
    def test_basic(self):
        # 3 subjects, 3 categories, 5 raters
        matrix = [[3, 2, 0], [0, 4, 1], [1, 1, 3]]
        result = fleiss_kappa(matrix, n_raters=5)
        assert -1 <= result.value <= 1
        assert result.interpretation in ("poor", "slight", "fair", "moderate", "substantial", "almost_perfect")
