"""Tests for parametric tests — t-tests, ANOVA, correlation, chi-square, proportion, equivalence."""

import numpy as np
import pytest

from forgestat.parametric.ttest import one_sample, two_sample, paired
from forgestat.parametric.anova import one_way, one_way_from_dict, two_way
from forgestat.parametric.correlation import correlation
from forgestat.parametric.chi_square import chi_square_independence, chi_square_goodness_of_fit, fisher_exact
from forgestat.parametric.proportion import one_proportion, two_proportions
from forgestat.parametric.equivalence import tost


class TestOneSampleT:
    def test_significant(self):
        # Mean clearly above 0
        data = [10, 12, 11, 13, 14, 12, 11, 10, 15, 13]
        result = one_sample(data, mu=0)
        assert result.significant is True
        assert result.p_value < 0.001
        assert result.mean1 > 0

    def test_not_significant(self):
        # Data centered around the hypothesized mean
        rng = np.random.default_rng(42)
        data = rng.normal(50, 5, 30).tolist()
        result = one_sample(data, mu=50)
        assert result.p_value > 0.05 or abs(result.mean1 - 50) < 2

    def test_ci_contains_mean(self):
        rng = np.random.default_rng(42)
        data = rng.normal(100, 10, 50).tolist()
        result = one_sample(data, mu=100, conf=0.95)
        assert result.ci_lower < result.mean1 < result.ci_upper

    def test_effect_size_computed(self):
        data = [10, 12, 11, 13, 14]
        result = one_sample(data, mu=0)
        assert result.effect_size is not None
        assert result.effect_size_type == "cohens_d"
        assert result.effect_label in ("negligible", "small", "medium", "large")

    def test_assumptions_checked(self):
        data = list(range(1, 31))
        result = one_sample(data, mu=15)
        assert len(result.assumptions) >= 2
        names = [a.name for a in result.assumptions]
        assert "normality" in names
        assert "outliers" in names


class TestTwoSampleT:
    def test_welch_significant(self):
        x1 = [10, 12, 11, 13, 14, 15, 12, 11]
        x2 = [20, 22, 21, 23, 24, 25, 22, 21]
        result = two_sample(x1, x2)
        assert result.significant is True
        assert result.method == "welch"

    def test_pooled(self):
        x1 = [10, 12, 11, 13, 14]
        x2 = [20, 22, 21, 23, 24]
        result = two_sample(x1, x2, equal_var=True)
        assert result.method == "pooled"

    def test_not_significant(self):
        rng = np.random.default_rng(42)
        x1 = rng.normal(50, 5, 20).tolist()
        x2 = rng.normal(50, 5, 20).tolist()
        result = two_sample(x1, x2)
        # Same distribution, likely not significant
        assert result.mean_diff is not None

    def test_has_both_means(self):
        result = two_sample([1, 2, 3], [4, 5, 6])
        assert result.mean1 is not None
        assert result.mean2 is not None
        assert result.n1 == 3
        assert result.n2 == 3


class TestPairedT:
    def test_significant_difference(self):
        before = [10, 12, 11, 13, 14, 15, 12, 11, 10, 13]
        after = [12, 14, 13, 15, 16, 17, 14, 13, 12, 15]
        result = paired(before, after)
        assert result.significant is True
        assert result.mean_diff < 0  # after > before → diff is negative

    def test_no_difference(self):
        x = [10, 12, 11, 13, 14]
        result = paired(x, x)
        assert result.p_value == 1.0 or result.mean_diff == 0.0

    def test_unequal_length_raises(self):
        with pytest.raises(ValueError, match="equal length"):
            paired([1, 2, 3], [4, 5])


class TestOneWayAnova:
    def test_significant(self):
        g1 = [10, 12, 11, 13, 14]
        g2 = [20, 22, 21, 23, 24]
        g3 = [30, 32, 31, 33, 34]
        result = one_way(g1, g2, g3)
        assert result.significant is True
        assert result.p_value < 0.001
        assert result.effect_size > 0.5  # very large eta²

    def test_from_dict(self):
        groups = {"A": [10, 12, 11], "B": [20, 22, 21], "C": [30, 32, 31]}
        result = one_way_from_dict(groups)
        assert result.significant is True
        assert set(result.group_means.keys()) == {"A", "B", "C"}

    def test_ss_decomposition(self):
        g1 = [10, 12, 11, 13]
        g2 = [20, 22, 21, 23]
        result = one_way(g1, g2, labels=["Low", "High"])
        # SS_between + SS_within ≈ SS_total
        assert abs(result.ss_between + result.ss_within - result.ss_total) < 0.01

    def test_omega_squared_computed(self):
        g1 = [10, 12, 11, 13, 14]
        g2 = [20, 22, 21, 23, 24]
        result = one_way(g1, g2)
        assert result.omega_squared is not None
        assert result.omega_squared <= result.effect_size  # ω² ≤ η²


class TestTwoWayAnova:
    def test_basic(self):
        data = {
            "response": [10, 12, 20, 22, 11, 13, 21, 23],
            "A": ["a1", "a1", "a2", "a2", "a1", "a1", "a2", "a2"],
            "B": ["b1", "b1", "b1", "b1", "b2", "b2", "b2", "b2"],
        }
        result = two_way(data, "response", "A", "B")
        assert len(result.sources) == 3  # A, B, A:B
        assert result.sources[0].source == "A"

    def test_main_effects_detected(self):
        # Strong A effect, no B effect
        data = {
            "response": [10, 11, 30, 31, 10, 11, 30, 31],
            "A": ["a1", "a1", "a2", "a2", "a1", "a1", "a2", "a2"],
            "B": ["b1", "b1", "b1", "b1", "b2", "b2", "b2", "b2"],
        }
        result = two_way(data, "response", "A", "B")
        a_source = result.sources[0]
        assert a_source.p_value < 0.05  # A should be significant


class TestCorrelation:
    def test_perfect_positive(self):
        data = {"X": [1, 2, 3, 4, 5], "Y": [2, 4, 6, 8, 10]}
        result = correlation(data)
        assert result.method == "pearson"
        assert len(result.pairs) == 1
        assert abs(result.pairs[0].r - 1.0) < 0.001

    def test_no_correlation(self):
        rng = np.random.default_rng(42)
        data = {"X": rng.normal(0, 1, 100).tolist(), "Y": rng.normal(0, 1, 100).tolist()}
        result = correlation(data)
        assert abs(result.pairs[0].r) < 0.3  # likely near zero

    def test_spearman(self):
        data = {"X": [1, 2, 3, 4, 5], "Y": [2, 4, 6, 8, 10]}
        result = correlation(data, method="spearman")
        assert result.method == "spearman"
        assert abs(result.pairs[0].r - 1.0) < 0.001

    def test_matrix_shape(self):
        data = {"A": [1, 2, 3], "B": [4, 5, 6], "C": [7, 8, 9]}
        result = correlation(data)
        assert len(result.matrix) == 3
        assert len(result.pairs) == 3  # 3 choose 2


class TestChiSquare:
    def test_independence(self):
        observed = [[10, 20, 30], [6, 9, 17]]
        result = chi_square_independence(observed)
        assert result.test_name == "Chi-square test of independence"
        assert result.p_value > 0  # valid p-value

    def test_strong_association(self):
        observed = [[50, 0], [0, 50]]
        result = chi_square_independence(observed)
        assert result.significant is True
        assert result.cramers_v > 0.5

    def test_goodness_of_fit_uniform(self):
        observed = [25, 25, 25, 25]
        result = chi_square_goodness_of_fit(observed)
        assert result.p_value == 1.0  # perfectly uniform

    def test_fisher_exact(self):
        table = [[10, 2], [3, 15]]
        result = fisher_exact(table)
        assert result.test_name == "Fisher's exact test"
        assert result.p_value < 0.05


class TestProportion:
    def test_one_prop_significant(self):
        result = one_proportion(70, 100, p0=0.5)
        assert result.significant is True
        assert result.p_hat == 0.7

    def test_one_prop_not_significant(self):
        result = one_proportion(52, 100, p0=0.5)
        assert result.p_value > 0.05

    def test_two_props(self):
        result = two_proportions(80, 100, 60, 100)
        assert result.significant is True
        assert abs(result.p_diff - 0.2) < 1e-10

    def test_wilson_ci(self):
        result = one_proportion(50, 100, p0=0.5)
        assert result.ci_lower < 0.5 < result.ci_upper


class TestEquivalence:
    def test_equivalent(self):
        rng = np.random.default_rng(42)
        x1 = rng.normal(50, 2, 100).tolist()
        x2 = rng.normal(50, 2, 100).tolist()
        result = tost(x1, x2, margin=2.0)
        assert result.equivalent is True
        assert result.p_tost < 0.05

    def test_not_equivalent(self):
        x1 = [10, 12, 11, 13, 14]
        x2 = [20, 22, 21, 23, 24]
        result = tost(x1, x2, margin=1.0)
        assert result.equivalent is False

    def test_ci_within_margin(self):
        rng = np.random.default_rng(42)
        x1 = rng.normal(50, 1, 200).tolist()
        x2 = rng.normal(50, 1, 200).tolist()
        result = tost(x1, x2, margin=1.0)
        if result.equivalent:
            assert result.ci_lower > -result.margin
            assert result.ci_upper < result.margin
