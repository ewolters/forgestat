"""Interpretation engine — structured plain English for any forgestat result.

Generates explanations, recommendations, validation reports, and method comparisons.
Deterministic logic only.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from ..core.effect_size import classify_effect
from ..core.types import (
    AnovaResult,
    ChiSquareResult,
    CorrelationResult,
    EquivalenceResult,
    PostHocResult,
    RankTestResult,
    TestResult,
    TTestResult,
)


@dataclass
class Explanation:
    """Structured explanation of a statistical result."""

    summary: str = ""  # 1-2 sentence headline
    detail: str = ""  # full paragraph
    assumptions: str = ""  # assumption status summary
    effect: str = ""  # effect size interpretation
    practical: str = ""  # practical significance note
    caveats: list[str] = field(default_factory=list)
    confidence: str = ""  # overall confidence rating


@dataclass
class Recommendation:
    """A recommended next step after an analysis."""

    action: str  # what to do
    reason: str  # why
    function: str  # forgestat function to call
    priority: str  # "required", "suggested", "optional"


@dataclass
class ValidationReport:
    """Assessment of how trustworthy a result is."""

    is_trustworthy: bool = True
    confidence_level: str = "moderate"  # high, moderate, low, unreliable
    issues: list[str] = field(default_factory=list)
    strengths: list[str] = field(default_factory=list)
    overall: str = ""


@dataclass
class ComparisonReport:
    """Result of running both parametric and non-parametric tests."""

    methods: list[dict] = field(default_factory=list)
    agreement: bool = True
    recommended: str = ""
    summary: str = ""


def explain(result: Any, context: str | None = None) -> Explanation:
    """Generate a structured explanation for any forgestat result type.

    Args:
        result: A forgestat result (TTestResult, AnovaResult, CorrelationResult, etc.)
        context: Optional context string to include in the explanation.

    Returns:
        Explanation with summary, detail, assumptions, effect, practical notes, caveats.
    """
    if isinstance(result, TTestResult):
        return _explain_ttest(result, context)
    elif isinstance(result, AnovaResult):
        return _explain_anova(result, context)
    elif isinstance(result, CorrelationResult):
        return _explain_correlation(result, context)
    elif isinstance(result, ChiSquareResult):
        return _explain_chi_square(result, context)
    elif isinstance(result, RankTestResult):
        return _explain_rank_test(result, context)
    elif isinstance(result, EquivalenceResult):
        return _explain_equivalence(result, context)
    elif isinstance(result, TestResult):
        return _explain_generic_test(result, context)
    else:
        return Explanation(
            summary=f"Result of type {type(result).__name__}",
            detail="No specific interpretation template available for this result type.",
            confidence="moderate",
        )


def recommend_next(result: Any, profile: Any = None) -> list[Recommendation]:
    """Generate recommendations for next steps after an analysis.

    Args:
        result: Any forgestat result.
        profile: Optional DataProfile for additional context.

    Returns:
        List of Recommendation objects, sorted by priority.
    """
    recs: list[Recommendation] = []

    # ANOVA significant -> post-hoc
    if isinstance(result, AnovaResult) and result.significant:
        recs.append(Recommendation(
            action="Run post-hoc pairwise comparisons to identify which groups differ",
            reason="ANOVA is significant but only tells you groups differ overall, not which specific pairs",
            function="posthoc.comparisons.tukey_hsd",
            priority="required",
        ))

    # Low power detection
    power = _estimate_power(result)
    if power is not None and power < 0.8:
        recs.append(Recommendation(
            action=f"Compute required sample size for 80% power (current power ~{power:.2f})",
            reason="Low statistical power increases risk of Type II error (missing real effects)",
            function="power.sample_size.power_t_test",
            priority="suggested",
        ))

    # Assumption violations -> non-parametric
    if hasattr(result, "assumptions"):
        for a in result.assumptions:
            if a.name == "normality" and not a.passed:
                if isinstance(result, TTestResult):
                    if result.method == "paired":
                        recs.append(Recommendation(
                            action="Run Wilcoxon signed-rank test as non-parametric alternative",
                            reason="Normality assumption is violated",
                            function="nonparametric.rank_tests.wilcoxon_signed_rank",
                            priority="suggested",
                        ))
                    else:
                        recs.append(Recommendation(
                            action="Run Mann-Whitney U test as non-parametric alternative",
                            reason="Normality assumption is violated",
                            function="nonparametric.rank_tests.mann_whitney",
                            priority="suggested",
                        ))
                elif isinstance(result, AnovaResult):
                    recs.append(Recommendation(
                        action="Run Kruskal-Wallis test as non-parametric alternative",
                        reason="Normality assumption is violated for one or more groups",
                        function="nonparametric.rank_tests.kruskal_wallis",
                        priority="suggested",
                    ))
                break  # only suggest once

    # Outliers detected -> robust alternative
    if hasattr(result, "assumptions"):
        for a in result.assumptions:
            if a.name == "outliers" and not a.passed:
                recs.append(Recommendation(
                    action="Investigate outliers and consider robust statistical methods",
                    reason="Significant outliers detected that may inflate or deflate the test statistic",
                    function="exploratory.univariate.describe",
                    priority="suggested",
                ))
                break

    # Correlation found -> suggest regression
    if isinstance(result, CorrelationResult):
        for pair in result.pairs:
            if pair.p_value < 0.05:
                recs.append(Recommendation(
                    action="Run regression analysis to model the predictive relationship",
                    reason=f"Significant correlation found between {pair.var1} and {pair.var2}",
                    function="regression.linear.ols",
                    priority="optional",
                ))
                break

    # Chi-square significant -> examine residuals
    if isinstance(result, ChiSquareResult) and result.significant:
        recs.append(Recommendation(
            action="Examine standardized residuals to identify which cells drive the association",
            reason="Chi-square is significant, residuals show where observed differs most from expected",
            function="parametric.chi_square.chi_square_independence",
            priority="suggested",
        ))

    # Multiple comparisons -> FDR correction
    if isinstance(result, AnovaResult) and result.significant:
        recs.append(Recommendation(
            action="Apply Benjamini-Hochberg FDR correction if running multiple tests",
            reason="Multiple comparisons inflate false discovery rate",
            function="intelligence.corrections.benjamini_hochberg",
            priority="optional",
        ))

    # Sort by priority
    priority_order = {"required": 0, "suggested": 1, "optional": 2}
    recs.sort(key=lambda r: priority_order.get(r.priority, 3))

    return recs


def validate_conclusion(result: Any, alpha: float = 0.05) -> ValidationReport:
    """Validate whether a statistical conclusion is trustworthy.

    Checks sample size, power, assumptions, effect/significance alignment, borderline p.

    Args:
        result: Any forgestat result with p_value.
        alpha: Significance level used.

    Returns:
        ValidationReport with issues, strengths, and overall assessment.
    """
    issues: list[str] = []
    strengths: list[str] = []

    # Sample size
    n_total = _get_sample_size(result)
    if n_total is not None:
        if n_total < 10:
            issues.append(f"Very small sample (n={n_total}) — results are unreliable")
        elif n_total < 20:
            issues.append(f"Small sample (n={n_total}) — limited statistical power")
        elif n_total >= 100:
            strengths.append(f"Good sample size (n={n_total})")
        else:
            strengths.append(f"Adequate sample size (n={n_total})")

    # Power
    power = _estimate_power(result)
    if power is not None:
        if power < 0.5:
            issues.append(f"Very low power ({power:.2f}) — high risk of missing real effects")
        elif power < 0.8:
            issues.append(f"Underpowered ({power:.2f}) — consider larger sample")
        else:
            strengths.append(f"Adequate power ({power:.2f})")

    # Assumption violations
    if hasattr(result, "assumptions"):
        n_violations = sum(1 for a in result.assumptions if not a.passed)
        n_checks = len(result.assumptions)
        if n_violations == 0 and n_checks > 0:
            strengths.append("All assumptions met")
        elif n_violations > 0:
            violated = [a.name for a in result.assumptions if not a.passed]
            issues.append(f"Assumption violations: {', '.join(violated)}")

    # P-value diagnostics
    p = getattr(result, "p_value", None) or getattr(result, "p_tost", None)
    if p is not None:
        # Borderline
        if 0.04 < p < 0.06:
            issues.append(f"Borderline p-value ({p:.4f}) — interpret with caution")
        # Effect size alignment
        es = getattr(result, "effect_size", None)
        if es is not None and isinstance(es, (int, float)):
            es_type = getattr(result, "effect_size_type", "cohens_d")
            mag = classify_effect(es, es_type)
            sig = p < alpha
            if sig and mag == "negligible":
                issues.append("Statistically significant but negligible effect size — may not be practically meaningful")
            elif not sig and mag in ("medium", "large"):
                issues.append(f"Not significant but {mag} effect size — sample may be too small")

    # Overall assessment
    if not issues:
        confidence_level = "high"
        is_trustworthy = True
        overall = "The analysis is well-supported with adequate data and met assumptions."
    elif len(issues) == 1:
        confidence_level = "moderate"
        is_trustworthy = True
        overall = f"The analysis is generally trustworthy with one concern: {issues[0].lower()}."
    elif len(issues) <= 3:
        confidence_level = "low"
        is_trustworthy = False
        overall = f"The analysis has {len(issues)} concerns that reduce confidence in the conclusion."
    else:
        confidence_level = "unreliable"
        is_trustworthy = False
        overall = "Multiple issues make this analysis unreliable. Consider collecting more data or using different methods."

    return ValidationReport(
        is_trustworthy=is_trustworthy,
        confidence_level=confidence_level,
        issues=issues,
        strengths=strengths,
        overall=overall,
    )


def compare_methods(
    data: dict[str, list[float]] | list,
    alpha: float = 0.05,
    paired: bool = False,
    labels: list[str] | None = None,
    **kwargs: Any,
) -> ComparisonReport:
    """Run both parametric and non-parametric tests on the same data.

    Args:
        data: Dict of groups or list of arrays.
        alpha: Significance level.
        paired: Whether data is paired.
        labels: Group names.

    Returns:
        ComparisonReport showing agreement/disagreement between methods.
    """
    from .engine import _normalize_groups

    groups = _normalize_groups(data, labels)
    names = list(groups.keys())
    arrays = [np.asarray(v, dtype=float) for v in groups.values()]
    arrays = [a[np.isfinite(a)] for a in arrays]
    k = len(arrays)

    methods: list[dict] = []

    if k == 1:
        mu = kwargs.get("mu", 0.0)
        # Parametric
        from ..parametric.ttest import one_sample
        try:
            para = one_sample(arrays[0].tolist(), mu=mu, alpha=alpha)
            methods.append({
                "name": "One-sample t-test",
                "test": "parametric",
                "result": para,
                "significant": para.significant,
                "p_value": para.p_value,
                "effect": _extract_es_dict(para),
            })
        except Exception:
            pass

        # Non-parametric (Wilcoxon)
        from ..nonparametric.rank_tests import wilcoxon_signed_rank
        shifted = (arrays[0] - mu).tolist()
        if len(shifted) >= 6:
            try:
                nonpara = wilcoxon_signed_rank(shifted, alpha=alpha)
                methods.append({
                    "name": "Wilcoxon signed-rank",
                    "test": "non-parametric",
                    "result": nonpara,
                    "significant": nonpara.significant,
                    "p_value": nonpara.p_value,
                    "effect": _extract_es_dict(nonpara),
                })
            except Exception:
                pass

    elif k == 2:
        if paired:
            # Paired t-test
            from ..parametric.ttest import paired as paired_ttest
            try:
                para = paired_ttest(arrays[0].tolist(), arrays[1].tolist(), alpha=alpha)
                methods.append({
                    "name": "Paired t-test",
                    "test": "parametric",
                    "result": para,
                    "significant": para.significant,
                    "p_value": para.p_value,
                    "effect": _extract_es_dict(para),
                })
            except Exception:
                pass

            # Wilcoxon signed-rank
            from ..nonparametric.rank_tests import wilcoxon_signed_rank
            if len(arrays[0]) >= 6:
                try:
                    nonpara = wilcoxon_signed_rank(arrays[0].tolist(), arrays[1].tolist(), alpha=alpha)
                    methods.append({
                        "name": "Wilcoxon signed-rank",
                        "test": "non-parametric",
                        "result": nonpara,
                        "significant": nonpara.significant,
                        "p_value": nonpara.p_value,
                        "effect": _extract_es_dict(nonpara),
                    })
                except Exception:
                    pass
        else:
            # Two-sample t-test (Welch)
            from ..parametric.ttest import two_sample
            try:
                para = two_sample(arrays[0].tolist(), arrays[1].tolist(), alpha=alpha)
                methods.append({
                    "name": para.test_name,
                    "test": "parametric",
                    "result": para,
                    "significant": para.significant,
                    "p_value": para.p_value,
                    "effect": _extract_es_dict(para),
                })
            except Exception:
                pass

            # Mann-Whitney U
            from ..nonparametric.rank_tests import mann_whitney
            try:
                nonpara = mann_whitney(arrays[0].tolist(), arrays[1].tolist(), alpha=alpha)
                methods.append({
                    "name": "Mann-Whitney U",
                    "test": "non-parametric",
                    "result": nonpara,
                    "significant": nonpara.significant,
                    "p_value": nonpara.p_value,
                    "effect": _extract_es_dict(nonpara),
                })
            except Exception:
                pass

    elif k >= 3:
        # ANOVA
        from ..parametric.anova import one_way
        try:
            para = one_way(*[a.tolist() for a in arrays], labels=names, alpha=alpha)
            methods.append({
                "name": "One-way ANOVA",
                "test": "parametric",
                "result": para,
                "significant": para.significant,
                "p_value": para.p_value,
                "effect": _extract_es_dict(para),
            })
        except Exception:
            pass

        # Kruskal-Wallis
        from ..nonparametric.rank_tests import kruskal_wallis
        try:
            nonpara = kruskal_wallis(*[a.tolist() for a in arrays], labels=names, alpha=alpha)
            methods.append({
                "name": "Kruskal-Wallis H",
                "test": "non-parametric",
                "result": nonpara,
                "significant": nonpara.significant,
                "p_value": nonpara.p_value,
                "effect": _extract_es_dict(nonpara),
            })
        except Exception:
            pass

    # Determine agreement
    sig_values = [m["significant"] for m in methods]
    agreement = len(set(sig_values)) <= 1 if sig_values else True

    # Recommendation
    if not methods:
        recommended = "Unable to run any tests"
        summary = "No tests could be executed on the provided data."
    elif agreement:
        recommended = methods[0]["name"]
        summary = (
            f"Both parametric and non-parametric methods agree "
            f"({'significant' if sig_values[0] else 'not significant'}). "
            f"Results are robust to method choice."
        )
    else:
        # Disagreement — recommend based on assumptions
        from ..core.assumptions import check_normality
        all_normal = True
        for arr in arrays:
            if len(arr) >= 3:
                chk = check_normality(arr)
                if not chk.passed:
                    all_normal = False
                    break

        if all_normal:
            recommended = next((m["name"] for m in methods if m["test"] == "parametric"), methods[0]["name"])
            summary = (
                "Methods disagree on significance. Data appears normal, "
                "so the parametric test is preferred (higher power)."
            )
        else:
            recommended = next((m["name"] for m in methods if m["test"] == "non-parametric"), methods[0]["name"])
            summary = (
                "Methods disagree on significance. Data departs from normality, "
                "so the non-parametric test is more trustworthy."
            )

    return ComparisonReport(
        methods=methods,
        agreement=agreement,
        recommended=recommended,
        summary=summary,
    )


# ---------------------------------------------------------------------------
# Internal explain helpers
# ---------------------------------------------------------------------------

def _explain_ttest(result: TTestResult, context: str | None) -> Explanation:
    """Explain a t-test result."""
    test_type = result.test_name
    sig = "significant" if result.significant else "no significant"
    p = result.p_value
    t = result.statistic
    caveats = []

    # Summary
    if result.mean2 is not None:
        summary = (
            f"The {test_type} found {sig} difference "
            f"(t={t:.3f}, p={p:.4f})."
        )
    else:
        summary = (
            f"The {test_type} found {sig} difference from the hypothesized mean "
            f"(t={t:.3f}, p={p:.4f})."
        )

    # Detail
    parts = [summary]
    if result.ci_lower is not None and result.ci_upper is not None:
        parts.append(f"95% CI for the difference: [{result.ci_lower:.3f}, {result.ci_upper:.3f}].")
    if result.mean2 is not None:
        parts.append(f"Group means: {result.mean1:.3f} vs {result.mean2:.3f} (diff = {result.mean_diff:.3f}).")
    else:
        parts.append(f"Sample mean: {result.mean1:.3f} (diff from reference = {result.mean_diff:.3f}).")
    detail = " ".join(parts)

    # Assumptions
    assumption_strs = []
    for a in result.assumptions:
        status = "PASS" if a.passed else "FAIL"
        assumption_strs.append(f"{a.name} ({a.test_name}): {status}")
    assumptions_text = "; ".join(assumption_strs) if assumption_strs else "No assumption checks available."

    # Effect size
    d = result.effect_size
    es_label = result.effect_label
    if d is not None:
        effect_text = f"Cohen's d = {d:.3f} ({es_label}), indicating a {es_label} practical effect."
    else:
        effect_text = "No effect size computed."

    # Practical significance
    if result.significant and es_label in ("negligible", "small"):
        practical = (
            "While statistically significant, the practical effect is small. "
            "Consider whether this difference matters in context."
        )
        caveats.append("Small effect despite statistical significance")
    elif result.significant and es_label in ("medium", "large"):
        practical = "This is both statistically and practically significant."
    elif not result.significant:
        practical = (
            "No statistically significant difference was found. "
            "This could mean no real difference exists, or the sample was too small to detect it."
        )
    else:
        practical = ""

    # Assumption caveats
    for a in result.assumptions:
        if not a.passed:
            caveats.append(f"{a.name}: {a.suggestion}" if a.suggestion else f"{a.name} assumption not met")

    # Borderline p
    if 0.04 < p < 0.06:
        caveats.append(f"Borderline p-value ({p:.4f}) — interpret with caution")

    confidence = "high"
    n = result.n1 + (result.n2 or 0)
    if n < 20:
        confidence = "moderate"
    if any(not a.passed for a in result.assumptions):
        confidence = "moderate"
    if n < 10:
        confidence = "low"

    return Explanation(
        summary=summary,
        detail=detail,
        assumptions=assumptions_text,
        effect=effect_text,
        practical=practical,
        caveats=caveats,
        confidence=confidence,
    )


def _explain_anova(result: AnovaResult, context: str | None) -> Explanation:
    """Explain an ANOVA result."""
    sig = "significant" if result.significant else "no significant"
    f = result.statistic
    p = result.p_value
    k = len(result.group_means)
    caveats = []

    summary = (
        f"The one-way ANOVA found {sig} difference among the {k} groups "
        f"(F({result.df_between:.0f},{result.df_within:.0f}) = {f:.3f}, p = {p:.4f})."
    )

    means_str = ", ".join(f"{n}: {m:.3f}" for n, m in result.group_means.items())
    detail = f"{summary} Group means: {means_str}."

    # Assumptions
    assumption_strs = []
    for a in result.assumptions:
        status = "PASS" if a.passed else "FAIL"
        assumption_strs.append(f"{a.name} ({a.test_name}): {status}")
    assumptions_text = "; ".join(assumption_strs) if assumption_strs else "No assumption checks available."

    # Effect
    eta2 = result.effect_size
    omega2 = result.omega_squared
    effect_text = f"eta-squared = {eta2:.3f} ({result.effect_label})"
    if omega2 is not None:
        effect_text += f", omega-squared = {omega2:.3f}"
    effect_text += "."

    # Practical
    if result.significant and result.effect_label in ("medium", "large"):
        practical = "This is both statistically and practically significant."
    elif result.significant and result.effect_label in ("negligible", "small"):
        practical = "Statistically significant but the effect size is small."
        caveats.append("Small effect despite statistical significance")
    else:
        practical = "No significant differences detected among groups."

    for a in result.assumptions:
        if not a.passed:
            caveats.append(f"{a.name} assumption violated")

    n_total = sum(result.group_ns.values())
    confidence = "high" if n_total >= 30 * k else "moderate"
    if any(n < 5 for n in result.group_ns.values()):
        confidence = "low"

    return Explanation(
        summary=summary,
        detail=detail,
        assumptions=assumptions_text,
        effect=effect_text,
        practical=practical,
        caveats=caveats,
        confidence=confidence,
    )


def _explain_correlation(result: CorrelationResult, context: str | None) -> Explanation:
    """Explain a correlation result."""
    if not result.pairs:
        return Explanation(summary="No correlation pairs computed.", confidence="low")

    top = result.pairs[0]
    r = top.r
    p = top.p_value
    direction = "positive" if r > 0 else "negative"
    mag = classify_effect(r, "r")
    sig = "significant" if p < 0.05 else "not significant"

    summary = (
        f"The {result.method} correlation between {top.var1} and {top.var2} "
        f"is r = {r:.3f} (p = {p:.4f}), which is {sig}."
    )

    detail = summary
    if top.ci_lower is not None:
        detail += f" 95% CI: [{top.ci_lower:.3f}, {top.ci_upper:.3f}]."
    detail += f" R-squared = {top.r_squared:.3f} ({top.r_squared * 100:.1f}% of variance explained)."

    effect_text = f"r = {r:.3f} ({mag} {direction} association)"
    practical = f"The {direction} correlation explains {top.r_squared * 100:.1f}% of variance."

    caveats = []
    if p >= 0.05:
        caveats.append("Correlation is not statistically significant")
    if 0.04 < p < 0.06:
        caveats.append(f"Borderline p-value ({p:.4f})")

    return Explanation(
        summary=summary,
        detail=detail,
        assumptions="",
        effect=effect_text,
        practical=practical,
        caveats=caveats,
        confidence="high" if top.n >= 30 else "moderate",
    )


def _explain_chi_square(result: ChiSquareResult, context: str | None) -> Explanation:
    """Explain a chi-square test result."""
    sig = "significant" if result.significant else "no significant"
    chi2 = result.statistic
    p = result.p_value
    v = result.cramers_v

    summary = (
        f"The {result.test_name} found {sig} association "
        f"(chi-sq = {chi2:.3f}, df = {result.df:.0f}, p = {p:.4f})."
    )

    detail = summary
    if v > 0:
        detail += f" Cramer's V = {v:.3f} ({result.effect_label} association)."

    effect_text = f"Cramer's V = {v:.3f} ({result.effect_label})" if v > 0 else "No effect size computed."

    if result.significant and result.effect_label in ("medium", "large"):
        practical = "The association is both statistically and practically meaningful."
    elif result.significant:
        practical = "Statistically significant but the association strength is weak."
    else:
        practical = "No significant association found."

    caveats = []
    # Check for low expected frequencies
    if result.expected:
        try:
            exp_arr = np.asarray(result.expected)
            low_expected = np.sum(exp_arr < 5)
            if low_expected > 0:
                caveats.append(f"{int(low_expected)} cell(s) have expected frequency < 5 — consider Fisher's exact test")
        except Exception:
            pass

    return Explanation(
        summary=summary,
        detail=detail,
        assumptions="",
        effect=effect_text,
        practical=practical,
        caveats=caveats,
        confidence="high",
    )


def _explain_rank_test(result: RankTestResult, context: str | None) -> Explanation:
    """Explain a non-parametric rank test result."""
    sig = "significant" if result.significant else "no significant"
    stat = result.statistic
    p = result.p_value

    summary = (
        f"The {result.test_name} found {sig} difference "
        f"(statistic = {stat:.3f}, p = {p:.4f})."
    )

    detail = summary
    if result.median1 is not None and result.median2 is not None:
        detail += f" Group medians: {result.median1:.3f} vs {result.median2:.3f}."
    if result.median_diff is not None:
        detail += f" Median difference: {result.median_diff:.3f}."

    es = result.effect_size
    es_type = result.effect_size_type
    if es is not None:
        label = result.effect_label
        effect_text = f"{es_type} = {es:.3f} ({label})"
    else:
        effect_text = "No effect size computed."

    if result.significant:
        practical = "There is a statistically significant difference between the groups."
    else:
        practical = "No significant difference detected."

    caveats = []
    if 0.04 < p < 0.06:
        caveats.append(f"Borderline p-value ({p:.4f})")

    return Explanation(
        summary=summary,
        detail=detail,
        assumptions="Non-parametric test — no distributional assumptions required.",
        effect=effect_text,
        practical=practical,
        caveats=caveats,
        confidence="moderate",
    )


def _explain_equivalence(result: EquivalenceResult, context: str | None) -> Explanation:
    """Explain a TOST equivalence result."""
    if result.equivalent:
        summary = f"Equivalence established (TOST p = {result.p_tost:.4f})."
        practical = f"The groups are equivalent within +/-{result.margin}."
    else:
        summary = f"Equivalence not established (TOST p = {result.p_tost:.4f})."
        practical = f"Cannot conclude equivalence within +/-{result.margin}."

    detail = (
        f"{summary} Mean difference = {result.mean_diff:.3f}, "
        f"{result.ci_level * 100:.0f}% CI: [{result.ci_lower:.3f}, {result.ci_upper:.3f}]. "
        f"Equivalence margin: +/-{result.margin}."
    )

    es = result.effect_size
    effect_text = f"Cohen's d = {es:.3f}" if es is not None else ""

    caveats = []
    for a in result.assumptions:
        if not a.passed:
            caveats.append(f"{a.name} assumption violated")

    return Explanation(
        summary=summary,
        detail=detail,
        assumptions="TOST assumes normality and independence.",
        effect=effect_text,
        practical=practical,
        caveats=caveats,
        confidence="high" if result.equivalent else "moderate",
    )


def _explain_generic_test(result: TestResult, context: str | None) -> Explanation:
    """Explain a generic test result."""
    sig = "significant" if result.significant else "not significant"

    summary = f"The {result.test_name} result is {sig} (p = {result.p_value:.4f})."

    detail = f"{summary} Test statistic = {result.statistic:.3f}."
    if result.df is not None:
        detail += f" df = {result.df:.1f}."

    es = result.effect_size
    if es is not None:
        effect_text = f"{result.effect_size_type} = {es:.3f} ({result.effect_label})"
    else:
        effect_text = ""

    return Explanation(
        summary=summary,
        detail=detail,
        assumptions="",
        effect=effect_text,
        practical="",
        caveats=[],
        confidence="moderate",
    )


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _estimate_power(result: Any) -> float | None:
    """Rough power estimate from result's sample size and effect size."""
    es = getattr(result, "effect_size", None)
    if es is None or not isinstance(es, (int, float)) or not math.isfinite(es):
        return None

    n = _get_sample_size(result)
    if n is None or n < 2:
        return None

    # Quick approximation using normal distribution
    from scipy import stats as sp_stats
    try:
        z_alpha = sp_stats.norm.ppf(0.975)
        ncp = abs(es) * math.sqrt(n)
        power = 1 - sp_stats.norm.cdf(z_alpha - ncp)
        return min(1.0, max(0.0, power))
    except Exception:
        return None


def _get_sample_size(result: Any) -> int | None:
    """Extract total sample size from a result."""
    if hasattr(result, "n1"):
        n2 = getattr(result, "n2", None) or 0
        return result.n1 + n2
    if hasattr(result, "group_ns") and result.group_ns:
        return sum(result.group_ns.values())
    if hasattr(result, "n"):
        return result.n
    if hasattr(result, "extra") and "n_total" in result.extra:
        return result.extra["n_total"]
    return None


def _extract_es_dict(result: Any) -> dict | None:
    """Extract effect size as dict from result."""
    if hasattr(result, "effect_size") and result.effect_size is not None:
        return {
            "name": getattr(result, "effect_size_type", "unknown"),
            "value": result.effect_size,
            "magnitude": getattr(result, "effect_label", ""),
        }
    return None
