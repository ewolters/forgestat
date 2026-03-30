"""Tests for non-parametric and post-hoc tests."""

import numpy as np

from forgestat.nonparametric.rank_tests import (
    friedman,
    kruskal_wallis,
    mann_whitney,
    mood_median,
    sign_test,
    wilcoxon_signed_rank,
)
from forgestat.posthoc.comparisons import bonferroni, dunnett, dunn, games_howell, tukey_hsd


class TestMannWhitney:
    def test_significant(self):
        x1 = [10, 12, 11, 13, 14, 15, 12, 11]
        x2 = [20, 22, 21, 23, 24, 25, 22, 21]
        result = mann_whitney(x1, x2)
        assert result.significant is True
        assert result.test_name == "Mann-Whitney U"

    def test_effect_size(self):
        x1 = [10, 12, 11, 13, 14]
        x2 = [20, 22, 21, 23, 24]
        result = mann_whitney(x1, x2)
        assert result.effect_size > 0
        assert result.effect_size_type == "rank_biserial"

    def test_hodges_lehmann(self):
        x1 = [1, 2, 3, 4, 5]
        x2 = [4, 5, 6, 7, 8]
        result = mann_whitney(x1, x2)
        assert result.median_diff is not None
        assert result.median_diff < 0  # x1 < x2


class TestKruskalWallis:
    def test_significant(self):
        g1 = [10, 12, 11, 13, 14]
        g2 = [20, 22, 21, 23, 24]
        g3 = [30, 32, 31, 33, 34]
        result = kruskal_wallis(g1, g2, g3)
        assert result.significant is True

    def test_labels(self):
        result = kruskal_wallis([1, 2, 3], [4, 5, 6], labels=["A", "B"])
        assert "A" in result.extra["group_medians"]


class TestWilcoxon:
    def test_significant(self):
        x1 = [10, 12, 11, 13, 14, 15, 12, 11, 10, 13]
        x2 = [12, 14, 13, 15, 16, 17, 14, 13, 12, 15]
        result = wilcoxon_signed_rank(x1, x2)
        assert result.significant is True

    def test_one_sample(self):
        data = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        result = wilcoxon_signed_rank(data)  # test against median=0
        assert result.significant is True


class TestFriedman:
    def test_significant(self):
        c1 = [10, 12, 11, 13, 14, 15, 12, 11]
        c2 = [20, 22, 21, 23, 24, 25, 22, 21]
        c3 = [30, 32, 31, 33, 34, 35, 32, 31]
        result = friedman(c1, c2, c3)
        assert result.significant is True
        assert "kendalls_w" in result.extra


class TestMoodMedian:
    def test_basic(self):
        g1 = [10, 12, 11, 13, 14]
        g2 = [20, 22, 21, 23, 24]
        result = mood_median(g1, g2)
        assert result.test_name == "Mood's median"
        assert "grand_median" in result.extra


class TestSignTest:
    def test_above_median(self):
        data = [10, 12, 11, 13, 14, 15, 12, 11, 10, 13]
        result = sign_test(data, median0=5)
        assert result.significant is True

    def test_centered(self):
        data = [4, 5, 6, 5, 4, 6, 5, 5, 4, 6]
        result = sign_test(data, median0=5)
        assert result.p_value > 0.05


class TestTukeyHSD:
    def test_all_different(self):
        g1 = [10, 12, 11, 13, 14]
        g2 = [20, 22, 21, 23, 24]
        g3 = [30, 32, 31, 33, 34]
        result = tukey_hsd(g1, g2, g3, labels=["Low", "Med", "High"])
        assert result.test_name == "tukey_hsd"
        assert len(result.comparisons) == 3  # 3 choose 2
        assert all(c.significant for c in result.comparisons)

    def test_group_means(self):
        g1 = [10, 12, 11]
        g2 = [20, 22, 21]
        result = tukey_hsd(g1, g2, labels=["A", "B"])
        assert "A" in result.group_means
        assert "B" in result.group_means


class TestGamesHowell:
    def test_unequal_variances(self):
        rng = np.random.default_rng(42)
        g1 = rng.normal(10, 1, 20).tolist()
        g2 = rng.normal(20, 5, 20).tolist()
        g3 = rng.normal(30, 10, 20).tolist()
        result = games_howell(g1, g2, g3)
        assert result.test_name == "games_howell"
        assert len(result.comparisons) == 3


class TestDunnett:
    def test_vs_control(self):
        control = [10, 12, 11, 13, 14]
        treat1 = [20, 22, 21, 23, 24]
        treat2 = [10, 11, 12, 13, 14]
        result = dunnett(control, treat1, treat2, treatment_names=["Drug A", "Drug B"])
        assert result.control_group == "Control"
        assert len(result.comparisons) == 2
        # Drug A should be significant (big difference)
        drug_a = [c for c in result.comparisons if c.group1 == "Drug A"][0]
        assert drug_a.significant is True


class TestDunn:
    def test_nonparametric_posthoc(self):
        g1 = [10, 12, 11, 13, 14]
        g2 = [20, 22, 21, 23, 24]
        g3 = [30, 32, 31, 33, 34]
        result = dunn(g1, g2, g3, labels=["A", "B", "C"])
        assert result.test_name == "dunn"
        assert result.correction == "bonferroni"
        assert len(result.comparisons) == 3


class TestBonferroni:
    def test_correction_applied(self):
        g1 = [10, 12, 11, 13, 14]
        g2 = [15, 17, 16, 18, 19]
        g3 = [10, 11, 12, 13, 14]
        result = bonferroni(g1, g2, g3)
        # p-values should be adjusted (multiplied by n_pairs=3)
        assert all(c.p_value <= 1.0 for c in result.comparisons)
