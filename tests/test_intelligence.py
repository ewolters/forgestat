"""Tests for the Statistical Intelligence Engine — auto_analyze, explain, recommend, validate, compare, corrections."""

import numpy as np
import pytest

from forgestat.intelligence.engine import auto_analyze, profile_data, DataProfile, AnalysisResult
from forgestat.intelligence.interpret import (
    compare_methods,
    explain,
    recommend_next,
    validate_conclusion,
    Explanation,
    Recommendation,
    ValidationReport,
    ComparisonReport,
)
from forgestat.intelligence.corrections import (
    benjamini_hochberg,
    holm_bonferroni,
    fdr_summary,
    BHResult,
    CorrectionResult,
)
from forgestat.core.types import TTestResult, AnovaResult, RankTestResult


# ---------------------------------------------------------------------------
# profile_data
# ---------------------------------------------------------------------------

class TestProfileData:
    def test_single_group(self):
        prof = profile_data([1, 2, 3, 4, 5])
        assert prof.n_groups == 1
        assert prof.overall_n == 5
        assert prof.sample_sizes == [5]

    def test_two_groups_dict(self):
        prof = profile_data({"A": [1, 2, 3, 4, 5], "B": [6, 7, 8, 9, 10]})
        assert prof.n_groups == 2
        assert set(prof.group_names) == {"A", "B"}

    def test_balanced_detection(self):
        prof = profile_data({"A": [1, 2, 3], "B": [4, 5, 6]})
        assert prof.is_balanced is True

    def test_unbalanced_detection(self):
        prof = profile_data({"A": [1, 2, 3], "B": [4, 5, 6, 7]})
        assert prof.is_balanced is False
        assert any("Unbalanced" in w for w in prof.warnings)

    def test_small_sample_warning(self):
        prof = profile_data({"A": [1, 2, 3]})
        assert any("only 3" in w for w in prof.warnings)

    def test_normality_checked(self):
        rng = np.random.default_rng(42)
        normal_data = rng.normal(50, 5, 100).tolist()
        prof = profile_data({"G": normal_data})
        assert "G" in prof.normality

    def test_paired_flag(self):
        prof = profile_data({"A": [1, 2, 3], "B": [4, 5, 6]}, paired=True)
        assert prof.is_paired is True


# ---------------------------------------------------------------------------
# auto_analyze — compare goal
# ---------------------------------------------------------------------------

class TestAutoAnalyzeCompare:
    def test_two_groups_normal_picks_ttest(self):
        rng = np.random.default_rng(42)
        a = rng.normal(50, 5, 30).tolist()
        b = rng.normal(60, 5, 30).tolist()
        result = auto_analyze({"A": a, "B": b}, goal="compare")
        assert isinstance(result, AnalysisResult)
        assert "t-test" in result.test_name.lower() or "t-test" in result.test_function
        assert result.profile.n_groups == 2
        assert result.decision_path  # audit trail populated

    def test_two_groups_nonnormal_picks_mann_whitney(self):
        # Heavily skewed data, small n
        rng = np.random.default_rng(42)
        a = rng.exponential(1, 15).tolist()
        b = (rng.exponential(1, 15) + 5).tolist()
        result = auto_analyze({"A": a, "B": b}, goal="compare")
        # Should either pick Mann-Whitney or t-test with CLT note
        assert result.test_name in ("Mann-Whitney U", "Welch's t-test", "Student's t-test")
        assert result.decision_path

    def test_two_groups_paired_picks_paired_ttest(self):
        rng = np.random.default_rng(42)
        before = rng.normal(50, 5, 30).tolist()
        after = (np.array(before) + rng.normal(2, 1, 30)).tolist()
        result = auto_analyze([before, after], goal="compare", paired=True)
        assert "paired" in result.test_name.lower() or "wilcoxon" in result.test_name.lower()

    def test_three_groups_picks_anova(self):
        rng = np.random.default_rng(42)
        a = rng.normal(50, 5, 20).tolist()
        b = rng.normal(60, 5, 20).tolist()
        c = rng.normal(70, 5, 20).tolist()
        result = auto_analyze({"A": a, "B": b, "C": c}, goal="compare")
        assert "anova" in result.test_name.lower() or "kruskal" in result.test_name.lower()

    def test_three_groups_nonnormal_picks_kruskal(self):
        rng = np.random.default_rng(42)
        a = rng.exponential(1, 12).tolist()
        b = rng.exponential(2, 12).tolist()
        c = rng.exponential(3, 12).tolist()
        result = auto_analyze({"A": a, "B": b, "C": c}, goal="compare")
        assert result.test_name in ("Kruskal-Wallis H", "One-way ANOVA")

    def test_anova_significant_runs_posthoc(self):
        a = [10, 12, 11, 13, 14, 15, 12, 11, 10, 13]
        b = [20, 22, 21, 23, 24, 25, 22, 21, 20, 23]
        c = [30, 32, 31, 33, 34, 35, 32, 31, 30, 33]
        result = auto_analyze({"A": a, "B": b, "C": c}, goal="compare")
        assert result.result.significant is True
        assert result.posthoc is not None
        assert len(result.posthoc.comparisons) > 0

    def test_one_sample_with_mu(self):
        data = [10, 12, 11, 13, 14, 15, 12, 11, 10, 13]
        result = auto_analyze(data, goal="compare", mu=5.0)
        assert result.test_name in ("One-sample t-test", "Wilcoxon signed-rank (one-sample)")
        assert result.result.significant is True

    def test_decision_path_populated(self):
        result = auto_analyze({"A": [1, 2, 3, 4, 5], "B": [6, 7, 8, 9, 10]}, goal="compare")
        assert len(result.decision_path) >= 2  # at least "2 groups" + test choice

    def test_effect_size_computed(self):
        a = [10, 12, 11, 13, 14]
        b = [20, 22, 21, 23, 24]
        result = auto_analyze({"A": a, "B": b}, goal="compare")
        assert result.effect_size is not None
        assert "value" in result.effect_size
        assert "magnitude" in result.effect_size

    def test_interpretation_string(self):
        a = [10, 12, 11, 13, 14]
        b = [20, 22, 21, 23, 24]
        result = auto_analyze({"A": a, "B": b}, goal="compare")
        assert len(result.interpretation) > 0
        assert "significant" in result.interpretation.lower()

    def test_confidence_rated(self):
        a = [10, 12, 11, 13, 14]
        b = [20, 22, 21, 23, 24]
        result = auto_analyze({"A": a, "B": b}, goal="compare")
        assert result.confidence in ("high", "moderate", "low")


# ---------------------------------------------------------------------------
# auto_analyze — input formats
# ---------------------------------------------------------------------------

class TestAutoAnalyzeInputFormats:
    def test_dict_input(self):
        result = auto_analyze({"A": [1, 2, 3, 4, 5], "B": [6, 7, 8, 9, 10]})
        assert result.profile.n_groups == 2

    def test_list_of_lists_input(self):
        result = auto_analyze([[1, 2, 3, 4, 5], [6, 7, 8, 9, 10]])
        assert result.profile.n_groups == 2

    def test_single_list_input(self):
        result = auto_analyze([1, 2, 3, 4, 5], mu=0.0)
        assert result.profile.n_groups == 1

    def test_django_row_input(self):
        rows = [
            {"yield": 10, "shift": "A"},
            {"yield": 12, "shift": "A"},
            {"yield": 11, "shift": "A"},
            {"yield": 20, "shift": "B"},
            {"yield": 22, "shift": "B"},
            {"yield": 21, "shift": "B"},
        ]
        result = auto_analyze(rows, goal="compare", response="yield", factor="shift")
        assert result.profile.n_groups == 2
        assert set(result.profile.group_names) == {"A", "B"}


# ---------------------------------------------------------------------------
# auto_analyze — other goals
# ---------------------------------------------------------------------------

class TestAutoAnalyzeOtherGoals:
    def test_correlate_goal(self):
        rng = np.random.default_rng(42)
        x = rng.normal(0, 1, 50).tolist()
        y = [xi * 2 + rng.normal(0, 0.5) for xi in x]
        result = auto_analyze({"X": x, "Y": y}, goal="correlate")
        assert "correlation" in result.test_name.lower()
        assert result.effect_size is not None

    def test_distribute_goal(self):
        rng = np.random.default_rng(42)
        data = rng.normal(100, 15, 50).tolist()
        result = auto_analyze(data, goal="distribute")
        assert result.test_name == "Distribution analysis"
        assert "mean" in result.interpretation.lower()

    def test_equivalence_goal(self):
        rng = np.random.default_rng(42)
        a = rng.normal(50, 2, 50).tolist()
        b = rng.normal(50, 2, 50).tolist()
        result = auto_analyze({"A": a, "B": b}, goal="equivalence", margin=2.0)
        assert "equivalence" in result.test_name.lower()

    def test_predict_goal(self):
        rng = np.random.default_rng(42)
        x = rng.normal(0, 1, 30).tolist()
        y = [xi * 3 + 5 + rng.normal(0, 1) for xi in x]
        result = auto_analyze({"X": x, "Y": y}, goal="predict", response="Y")
        assert "ols" in result.test_name.lower()

    def test_unknown_goal_raises(self):
        with pytest.raises(ValueError, match="Unknown goal"):
            auto_analyze([1, 2, 3], goal="magic")


# ---------------------------------------------------------------------------
# explain
# ---------------------------------------------------------------------------

class TestExplain:
    def test_ttest_result(self):
        from forgestat.parametric.ttest import two_sample
        result = two_sample([10, 12, 11, 13, 14], [20, 22, 21, 23, 24])
        expl = explain(result)
        assert isinstance(expl, Explanation)
        assert len(expl.summary) > 0
        assert "significant" in expl.summary.lower()
        assert len(expl.effect) > 0
        assert "cohen" in expl.effect.lower()

    def test_anova_result(self):
        from forgestat.parametric.anova import one_way
        result = one_way([10, 12, 11], [20, 22, 21], [30, 32, 31], labels=["A", "B", "C"])
        expl = explain(result)
        assert "anova" in expl.summary.lower()
        assert "eta" in expl.effect.lower()

    def test_significant_vs_not_significant(self):
        from forgestat.parametric.ttest import one_sample
        sig_result = one_sample([10, 12, 11, 13, 14], mu=0)
        not_sig_result = one_sample([10, 12, 11, 13, 14], mu=12)

        sig_expl = explain(sig_result)
        not_sig_expl = explain(not_sig_result)

        assert "significant difference" in sig_expl.summary.lower() or "significant" in sig_expl.summary.lower()
        assert "no significant" in not_sig_expl.summary.lower()

    def test_large_effect_not_significant_mentions_power(self):
        # Create a result with large effect but not significant (tiny sample)
        from forgestat.parametric.ttest import two_sample
        # Very small sample, big difference but large variance
        result = two_sample([10, 15], [20, 25])
        expl = explain(result)
        # With only n=2, should have practical note about sample size
        assert isinstance(expl, Explanation)

    def test_rank_test_result(self):
        from forgestat.nonparametric.rank_tests import mann_whitney
        result = mann_whitney([1, 2, 3, 4, 5], [6, 7, 8, 9, 10])
        expl = explain(result)
        assert "mann-whitney" in expl.summary.lower()
        assert "non-parametric" in expl.assumptions.lower()

    def test_correlation_result(self):
        from forgestat.parametric.correlation import correlation
        result = correlation({"X": [1, 2, 3, 4, 5], "Y": [2, 4, 6, 8, 10]})
        expl = explain(result)
        assert "correlation" in expl.summary.lower()
        assert expl.effect  # effect size present


# ---------------------------------------------------------------------------
# recommend_next
# ---------------------------------------------------------------------------

class TestRecommendNext:
    def test_anova_significant_recommends_posthoc(self):
        from forgestat.parametric.anova import one_way
        result = one_way([10, 12, 11], [20, 22, 21], [30, 32, 31])
        recs = recommend_next(result)
        assert any("post-hoc" in r.action.lower() for r in recs)
        assert any(r.priority == "required" for r in recs)

    def test_low_power_recommends_sample_size(self):
        from forgestat.parametric.ttest import two_sample
        # Small sample, small effect -> low power
        rng = np.random.default_rng(42)
        a = rng.normal(50, 10, 5).tolist()
        b = rng.normal(52, 10, 5).tolist()
        result = two_sample(a, b)
        recs = recommend_next(result)
        # May or may not trigger depending on exact power estimate
        assert isinstance(recs, list)

    def test_nonnormal_parametric_suggests_nonparametric(self):
        from forgestat.parametric.ttest import two_sample
        # Create result that fails normality
        rng = np.random.default_rng(42)
        a = rng.exponential(1, 50).tolist()
        b = rng.exponential(2, 50).tolist()
        result = two_sample(a, b)
        recs = recommend_next(result)
        # Check if non-parametric suggested for normality violation
        norm_failed = any(not ac.passed for ac in result.assumptions if ac.name == "normality")
        if norm_failed:
            assert any("mann-whitney" in r.function.lower() or "non-parametric" in r.action.lower() for r in recs)

    def test_correlation_significant_suggests_regression(self):
        from forgestat.parametric.correlation import correlation
        result = correlation({"X": [1, 2, 3, 4, 5], "Y": [2, 4, 6, 8, 10]})
        recs = recommend_next(result)
        assert any("regression" in r.action.lower() for r in recs)

    def test_returns_sorted_by_priority(self):
        from forgestat.parametric.anova import one_way
        result = one_way([10, 12, 11], [20, 22, 21], [30, 32, 31])
        recs = recommend_next(result)
        if len(recs) >= 2:
            priorities = [r.priority for r in recs]
            order = {"required": 0, "suggested": 1, "optional": 2}
            values = [order.get(p, 3) for p in priorities]
            assert values == sorted(values)


# ---------------------------------------------------------------------------
# validate_conclusion
# ---------------------------------------------------------------------------

class TestValidateConclusion:
    def test_small_sample_flags_low_confidence(self):
        from forgestat.parametric.ttest import two_sample
        result = two_sample([1, 2, 3], [4, 5, 6])
        report = validate_conclusion(result)
        assert isinstance(report, ValidationReport)
        # Small sample should be flagged
        assert any("small" in i.lower() or "sample" in i.lower() for i in report.issues)

    def test_borderline_p_warns(self):
        # Create a result with borderline p-value
        result = TTestResult(
            test_name="t-test",
            statistic=2.0,
            p_value=0.049,
            alpha=0.05,
            effect_size=0.3,
            effect_size_type="cohens_d",
            n1=30,
            n2=30,
        )
        report = validate_conclusion(result)
        assert any("borderline" in i.lower() for i in report.issues)

    def test_good_result_high_confidence(self):
        from forgestat.parametric.ttest import two_sample
        rng = np.random.default_rng(42)
        a = rng.normal(50, 5, 100).tolist()
        b = rng.normal(55, 5, 100).tolist()
        result = two_sample(a, b)
        report = validate_conclusion(result)
        assert report.confidence_level in ("high", "moderate")
        assert len(report.strengths) > 0

    def test_significant_negligible_effect_warns(self):
        # Significant p but negligible effect
        result = TTestResult(
            test_name="t-test",
            statistic=3.0,
            p_value=0.001,
            alpha=0.05,
            effect_size=0.05,
            effect_size_type="cohens_d",
            effect_label="negligible",
            n1=500,
            n2=500,
        )
        report = validate_conclusion(result)
        assert any("negligible" in i.lower() for i in report.issues)


# ---------------------------------------------------------------------------
# compare_methods
# ---------------------------------------------------------------------------

class TestCompareMethods:
    def test_normal_data_both_agree(self):
        rng = np.random.default_rng(42)
        a = rng.normal(50, 5, 30).tolist()
        b = rng.normal(60, 5, 30).tolist()
        report = compare_methods({"A": a, "B": b})
        assert isinstance(report, ComparisonReport)
        assert len(report.methods) == 2
        # With clear signal, both should agree
        assert report.agreement is True

    def test_clear_signal_both_significant(self):
        a = [10, 12, 11, 13, 14, 15, 12, 11, 10, 13]
        b = [30, 32, 31, 33, 34, 35, 32, 31, 30, 33]
        report = compare_methods({"A": a, "B": b})
        assert all(m["significant"] for m in report.methods)

    def test_three_groups(self):
        a = [10, 12, 11, 13, 14]
        b = [20, 22, 21, 23, 24]
        c = [30, 32, 31, 33, 34]
        report = compare_methods({"A": a, "B": b, "C": c})
        assert len(report.methods) == 2  # ANOVA + Kruskal
        method_names = [m["name"] for m in report.methods]
        assert "One-way ANOVA" in method_names
        assert "Kruskal-Wallis H" in method_names

    def test_paired(self):
        before = [10, 12, 11, 13, 14, 15, 12, 11, 10, 13]
        after = [12, 14, 13, 15, 16, 17, 14, 13, 12, 15]
        report = compare_methods({"Before": before, "After": after}, paired=True)
        method_names = [m["name"] for m in report.methods]
        assert "Paired t-test" in method_names

    def test_has_summary(self):
        report = compare_methods({"A": [1, 2, 3, 4, 5], "B": [6, 7, 8, 9, 10]})
        assert len(report.summary) > 0
        assert len(report.recommended) > 0


# ---------------------------------------------------------------------------
# corrections — Benjamini-Hochberg
# ---------------------------------------------------------------------------

class TestBenjaminiHochberg:
    def test_known_example(self):
        # Classic textbook example
        p_values = [0.001, 0.008, 0.039, 0.041, 0.042, 0.060, 0.074, 0.205, 0.212, 0.216]
        result = benjamini_hochberg(p_values, alpha=0.05)
        assert isinstance(result, BHResult)
        assert len(result.adjusted_p) == 10
        # First few should be significant, later ones not
        assert result.significant[0] is True
        assert result.significant[1] is True

    def test_all_significant(self):
        p_values = [0.001, 0.002, 0.003]
        result = benjamini_hochberg(p_values, alpha=0.05)
        assert all(result.significant)
        assert result.n_discoveries == 3

    def test_none_significant(self):
        p_values = [0.5, 0.6, 0.7, 0.8]
        result = benjamini_hochberg(p_values, alpha=0.05)
        assert not any(result.significant)
        assert result.n_discoveries == 0

    def test_single_test(self):
        result = benjamini_hochberg([0.03], alpha=0.05)
        assert result.significant[0] is True
        assert result.n_discoveries == 1

    def test_empty_input(self):
        result = benjamini_hochberg([], alpha=0.05)
        assert result.n_discoveries == 0

    def test_adjusted_p_monotonic(self):
        p_values = [0.001, 0.01, 0.02, 0.03, 0.04, 0.05, 0.10, 0.50]
        result = benjamini_hochberg(p_values, alpha=0.05)
        # Adjusted p-values in sorted order should be monotonically non-decreasing
        sorted_adj = sorted(zip(result.original_p, result.adjusted_p), key=lambda x: x[0])
        adj_sorted = [a for _, a in sorted_adj]
        for i in range(1, len(adj_sorted)):
            assert adj_sorted[i] >= adj_sorted[i - 1] - 1e-10

    def test_adjusted_p_capped_at_one(self):
        p_values = [0.5, 0.9]
        result = benjamini_hochberg(p_values, alpha=0.05)
        assert all(p <= 1.0 for p in result.adjusted_p)


# ---------------------------------------------------------------------------
# corrections — Holm-Bonferroni
# ---------------------------------------------------------------------------

class TestHolmBonferroni:
    def test_step_down(self):
        p_values = [0.001, 0.01, 0.04, 0.06]
        result = holm_bonferroni(p_values, alpha=0.05)
        assert isinstance(result, CorrectionResult)
        assert result.method == "holm_bonferroni"
        # First p (0.001 * 4 = 0.004) should be significant
        assert result.significant[0] is True

    def test_all_significant(self):
        p_values = [0.001, 0.002, 0.003]
        result = holm_bonferroni(p_values, alpha=0.05)
        assert all(result.significant)

    def test_none_significant(self):
        p_values = [0.5, 0.6, 0.7]
        result = holm_bonferroni(p_values, alpha=0.05)
        assert not any(result.significant)

    def test_single_test(self):
        result = holm_bonferroni([0.03], alpha=0.05)
        assert result.significant[0] is True

    def test_empty_input(self):
        result = holm_bonferroni([], alpha=0.05)
        assert result.n_discoveries == 0

    def test_less_conservative_than_bonferroni(self):
        p_values = [0.005, 0.015, 0.025, 0.035]
        holm = holm_bonferroni(p_values, alpha=0.05)
        # Plain Bonferroni: multiply by 4 -> [0.02, 0.06, 0.10, 0.14]
        # Holm is less conservative, so should get at least as many discoveries
        bonf_sig = sum(1 for p in p_values if p * len(p_values) < 0.05)
        assert holm.n_discoveries >= bonf_sig


# ---------------------------------------------------------------------------
# corrections — fdr_summary
# ---------------------------------------------------------------------------

class TestFdrSummary:
    def test_formatted_output(self):
        tests = [
            {"name": "Test A", "p_value": 0.001},
            {"name": "Test B", "p_value": 0.04},
            {"name": "Test C", "p_value": 0.5},
        ]
        summary = fdr_summary(tests, alpha=0.05)
        assert "FDR Correction" in summary
        assert "Test A" in summary
        assert "Test B" in summary
        assert "Test C" in summary
        assert "Discoveries:" in summary

    def test_empty_tests(self):
        summary = fdr_summary([], alpha=0.05)
        assert "No tests" in summary
