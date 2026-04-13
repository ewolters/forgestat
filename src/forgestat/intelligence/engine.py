"""Statistical Intelligence Engine — automated test selection and execution.

Pure Python, deterministic decision logic. No AI/API calls.
Profiles data, selects the right test, runs it, and packages a complete result.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from scipy import stats

from ..core.assumptions import check_equal_variance, check_normality, check_outliers
from ..core.effect_size import (
    classify_effect,
    cohens_d_one_sample,
    cohens_d_paired,
    cohens_d_two_sample,
)
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
class DataProfile:
    """Profile of one or more datasets before analysis."""

    n_groups: int = 0
    group_names: list[str] = field(default_factory=list)
    sample_sizes: list[int] = field(default_factory=list)
    data_type: str = "continuous"  # continuous, categorical, binary, ordinal, counts
    is_paired: bool = False
    is_balanced: bool = True
    normality: dict[str, bool] = field(default_factory=dict)
    equal_variance: bool = True
    outlier_counts: dict[str, int] = field(default_factory=dict)
    has_ties: bool = False
    overall_n: int = 0
    warnings: list[str] = field(default_factory=list)


@dataclass
class AnalysisResult:
    """Complete result of an automated analysis."""

    profile: DataProfile
    test_name: str  # human-readable test name
    test_function: str  # e.g. "parametric.ttest.two_sample"
    result: Any  # the actual TestResult/AnovaResult/etc.
    effect_size: dict | None = None  # {name, value, magnitude}
    assumptions: list[dict] = field(default_factory=list)  # [{check, passed, detail, action_taken}]
    posthoc: Any | None = None  # PostHocResult if applicable
    power: dict | None = None  # {power, achieved_n, required_n_for_80}
    interpretation: str = ""  # plain English summary
    recommendations: list[str] = field(default_factory=list)
    confidence: str = "moderate"  # high, moderate, low
    decision_path: list[str] = field(default_factory=list)  # audit trail


def profile_data(
    data: dict[str, list[float]] | list | np.ndarray,
    labels: list[str] | None = None,
    paired: bool = False,
    alpha: float = 0.05,
) -> DataProfile:
    """Profile one or more datasets before analysis.

    Args:
        data: Dict of {group_name: values}, list of arrays, or single array.
        labels: Group names (if data is list of arrays).
        paired: Whether groups are paired/repeated measures.
        alpha: Significance level for assumption checks.

    Returns:
        DataProfile with normality, variance, outlier, and balance info.
    """
    groups = _normalize_groups(data, labels)
    names = list(groups.keys())
    arrays = [np.asarray(v, dtype=float) for v in groups.values()]
    arrays = [a[np.isfinite(a)] for a in arrays]
    k = len(arrays)

    sizes = [len(a) for a in arrays]
    overall_n = sum(sizes)

    # Normality per group
    normality = {}
    for name, arr in zip(names, arrays):
        if len(arr) >= 3:
            chk = check_normality(arr, label=name, alpha=alpha)
            normality[name] = chk.passed
        else:
            normality[name] = True  # too few to test

    # Equal variance (2+ groups)
    equal_var = True
    if k >= 2 and all(len(a) >= 2 for a in arrays):
        var_check = check_equal_variance(*arrays, labels=names, alpha=alpha)
        equal_var = var_check.passed

    # Outlier counts
    outlier_counts = {}
    for name, arr in zip(names, arrays):
        if len(arr) >= 4:
            chk = check_outliers(arr, label=name)
            # Parse count from detail string
            detail = chk.detail
            try:
                count_str = detail.split(":")[1].strip().split("/")[0].strip()
                outlier_counts[name] = int(count_str)
            except (IndexError, ValueError):
                outlier_counts[name] = 0
        else:
            outlier_counts[name] = 0

    # Ties
    has_ties = False
    for arr in arrays:
        if len(arr) > 1:
            unique = len(np.unique(arr))
            if unique < len(arr):
                has_ties = True
                break

    # Balance
    is_balanced = len(set(sizes)) <= 1 if k > 1 else True

    # Data type heuristic
    data_type = _infer_data_type(arrays)

    # Warnings
    warnings = []
    for name, sz in zip(names, sizes):
        if sz < 5:
            warnings.append(f"{name} has only {sz} observations")
        elif sz < 20:
            warnings.append(f"{name} has {sz} observations — limited power")
    if not is_balanced and k > 1:
        warnings.append(f"Unbalanced design: group sizes {sizes}")

    return DataProfile(
        n_groups=k,
        group_names=names,
        sample_sizes=sizes,
        data_type=data_type,
        is_paired=paired,
        is_balanced=is_balanced,
        normality=normality,
        equal_variance=equal_var,
        outlier_counts=outlier_counts,
        has_ties=has_ties,
        overall_n=overall_n,
        warnings=warnings,
    )


def auto_analyze(
    data: dict[str, list[float]] | list | np.ndarray,
    goal: str = "compare",
    alpha: float = 0.05,
    paired: bool = False,
    labels: list[str] | None = None,
    mu: float | None = None,
    margin: float | None = None,
    response: str | None = None,
    factor: str | None = None,
    **kwargs: Any,
) -> AnalysisResult:
    """Automated statistical analysis. Takes data + intent, runs the right test.

    Args:
        data: Dict of groups, list of arrays, single array, or list of row dicts.
        goal: "compare", "correlate", "predict", "distribute", "equivalence".
        alpha: Significance level.
        paired: Whether groups are paired.
        labels: Group names.
        mu: Hypothesized mean (for one-sample tests).
        margin: Equivalence margin (for TOST).
        response: Response variable name (for row-dict input).
        factor: Factor variable name (for row-dict input).
        **kwargs: Passed to underlying test functions.

    Returns:
        AnalysisResult with test result, interpretation, recommendations, audit trail.
    """
    decision_path: list[str] = []

    # Handle row-dict input (Django style)
    if response is not None and factor is not None and isinstance(data, list):
        data = _rows_to_groups(data, response, factor)
        decision_path.append(f"Converted row dicts to groups by '{factor}'")

    # Dispatch by goal
    if goal == "compare":
        return _analyze_compare(data, alpha, paired, labels, mu, decision_path, **kwargs)
    elif goal == "correlate":
        return _analyze_correlate(data, alpha, labels, decision_path, **kwargs)
    elif goal == "predict":
        return _analyze_predict(data, alpha, labels, decision_path, **kwargs)
    elif goal == "distribute":
        return _analyze_distribute(data, alpha, labels, decision_path, **kwargs)
    elif goal == "equivalence":
        if margin is None:
            raise ValueError("Equivalence testing requires a margin parameter")
        return _analyze_equivalence(data, alpha, margin, labels, decision_path, **kwargs)
    else:
        raise ValueError(f"Unknown goal: {goal}. Use 'compare', 'correlate', 'predict', 'distribute', or 'equivalence'.")


# ---------------------------------------------------------------------------
# Goal: compare
# ---------------------------------------------------------------------------

def _analyze_compare(
    data: Any,
    alpha: float,
    paired: bool,
    labels: list[str] | None,
    mu: float | None,
    path: list[str],
    **kwargs: Any,
) -> AnalysisResult:
    """Route comparison analysis to the right test."""
    groups = _normalize_groups(data, labels)
    prof = profile_data(groups, paired=paired, alpha=alpha)
    k = prof.n_groups
    names = prof.group_names
    arrays = [np.asarray(groups[n], dtype=float) for n in names]
    arrays = [a[np.isfinite(a)] for a in arrays]

    # One-sample test
    if k == 1:
        if mu is None:
            mu = 0.0
        path.append(f"1 group, testing against mu={mu}")
        return _one_sample(arrays[0], mu, alpha, prof, path)

    # Two-group comparison
    if k == 2:
        path.append("2 groups")
        if paired:
            path.append("paired design")
            return _two_sample_paired(arrays[0], arrays[1], names, alpha, prof, path)
        else:
            path.append("independent samples")
            return _two_sample_independent(arrays[0], arrays[1], names, alpha, prof, path)

    # Three or more groups
    path.append(f"{k} groups")
    if paired:
        path.append("repeated measures design")
        return _multi_group_paired(arrays, names, alpha, prof, path)
    else:
        path.append("independent groups")
        return _multi_group_independent(arrays, names, alpha, prof, path)


def _one_sample(
    x: np.ndarray, mu: float, alpha: float, prof: DataProfile, path: list[str],
) -> AnalysisResult:
    """One-sample: t-test or Wilcoxon."""
    name = prof.group_names[0]
    is_normal = prof.normality.get(name, True)
    n = len(x)
    assumptions = []

    if is_normal or n >= 30:
        if not is_normal and n >= 30:
            path.append(f"non-normal but n={n} >= 30, using t-test (CLT)")
            assumptions.append({
                "check": "normality", "passed": False,
                "detail": "Data departs from normality but CLT applies (n >= 30)",
                "action_taken": "Proceeded with t-test due to large sample",
            })
        else:
            path.append("normal data -> one-sample t-test")

        from ..parametric.ttest import one_sample
        result = one_sample(x.tolist(), mu=mu, alpha=alpha)
        test_name = "One-sample t-test"
        test_func = "parametric.ttest.one_sample"
        es = _extract_effect_size(result)

        for a in result.assumptions:
            assumptions.append({
                "check": a.name, "passed": a.passed,
                "detail": a.detail, "action_taken": a.suggestion or "None needed",
            })

    else:
        path.append(f"non-normal, n={n} < 30 -> Wilcoxon signed-rank")
        from ..nonparametric.rank_tests import wilcoxon_signed_rank
        # Wilcoxon signed-rank against mu: test x - mu
        shifted = (x - mu).tolist()
        if len(shifted) < 6:
            # Fall back to t-test for very small samples
            path.append("n < 6, falling back to one-sample t-test")
            from ..parametric.ttest import one_sample
            result = one_sample(x.tolist(), mu=mu, alpha=alpha)
            test_name = "One-sample t-test"
            test_func = "parametric.ttest.one_sample"
        else:
            result = wilcoxon_signed_rank(shifted, alpha=alpha)
            test_name = "Wilcoxon signed-rank (one-sample)"
            test_func = "nonparametric.rank_tests.wilcoxon_signed_rank"
        es = _extract_effect_size(result)
        assumptions.append({
            "check": "normality", "passed": False,
            "detail": "Normality not met, used non-parametric alternative",
            "action_taken": "Switched to Wilcoxon signed-rank test",
        })

    power_info = _compute_power_one_sample(x, mu, alpha)
    interpretation = _interpret_one_sample(result, mu, es, power_info, test_name)
    recs = _recommendations_one_sample(result, prof, power_info)
    confidence = _rate_confidence(prof, power_info, assumptions)

    return AnalysisResult(
        profile=prof,
        test_name=test_name,
        test_function=test_func,
        result=result,
        effect_size=es,
        assumptions=assumptions,
        power=power_info,
        interpretation=interpretation,
        recommendations=recs,
        confidence=confidence,
        decision_path=path,
    )


def _two_sample_independent(
    x1: np.ndarray, x2: np.ndarray, names: list[str],
    alpha: float, prof: DataProfile, path: list[str],
) -> AnalysisResult:
    """Two independent groups: t-test or Mann-Whitney."""
    n1, n2 = len(x1), len(x2)
    normal1 = prof.normality.get(names[0], True)
    normal2 = prof.normality.get(names[1], True)
    both_normal = normal1 and normal2
    assumptions = []

    if both_normal or (n1 >= 30 and n2 >= 30):
        if not both_normal:
            path.append(f"non-normal but n1={n1}, n2={n2} >= 30, using t-test (CLT)")
            assumptions.append({
                "check": "normality", "passed": False,
                "detail": "One or both groups depart from normality but CLT applies",
                "action_taken": "Proceeded with t-test due to large samples",
            })
        else:
            path.append("checked normality -> normal")

        # Check equal variance to decide Welch vs Student
        if prof.equal_variance:
            path.append("equal variance -> Student's t-test")
            equal_var = True
        else:
            path.append("unequal variance -> Welch's t-test")
            equal_var = False
            assumptions.append({
                "check": "equal_variance", "passed": False,
                "detail": "Levene's test indicates unequal variances",
                "action_taken": "Using Welch's correction",
            })

        from ..parametric.ttest import two_sample
        result = two_sample(x1.tolist(), x2.tolist(), equal_var=equal_var, alpha=alpha)
        test_name = result.test_name
        test_func = "parametric.ttest.two_sample"

        for a in result.assumptions:
            assumptions.append({
                "check": a.name, "passed": a.passed,
                "detail": a.detail, "action_taken": a.suggestion or "None needed",
            })

    else:
        path.append(f"non-normal (n1={n1}, n2={n2}) -> Mann-Whitney U")
        from ..nonparametric.rank_tests import mann_whitney
        result = mann_whitney(x1.tolist(), x2.tolist(), alpha=alpha)
        test_name = "Mann-Whitney U"
        test_func = "nonparametric.rank_tests.mann_whitney"
        assumptions.append({
            "check": "normality", "passed": False,
            "detail": "Normality not met, used non-parametric alternative",
            "action_taken": "Switched to Mann-Whitney U test",
        })

    es = _extract_effect_size(result)
    power_info = _compute_power_two_sample(x1, x2, alpha)
    interpretation = _interpret_two_sample(result, names, es, power_info, test_name)
    recs = _recommendations_two_sample(result, prof, power_info)
    confidence = _rate_confidence(prof, power_info, assumptions)

    return AnalysisResult(
        profile=prof,
        test_name=test_name,
        test_function=test_func,
        result=result,
        effect_size=es,
        assumptions=assumptions,
        power=power_info,
        interpretation=interpretation,
        recommendations=recs,
        confidence=confidence,
        decision_path=path,
    )


def _two_sample_paired(
    x1: np.ndarray, x2: np.ndarray, names: list[str],
    alpha: float, prof: DataProfile, path: list[str],
) -> AnalysisResult:
    """Two paired groups: paired t-test or Wilcoxon signed-rank."""
    diff = x1 - x2
    diff_clean = diff[np.isfinite(diff)]
    n = len(diff_clean)
    assumptions = []

    # Check normality of differences
    normal_diff = True
    if n >= 3:
        norm_check = check_normality(diff_clean, label="Differences")
        normal_diff = norm_check.passed

    if normal_diff or n >= 30:
        if not normal_diff:
            path.append(f"differences non-normal but n={n} >= 30, using paired t-test (CLT)")
            assumptions.append({
                "check": "normality_of_differences", "passed": False,
                "detail": "Differences depart from normality but CLT applies",
                "action_taken": "Proceeded with paired t-test",
            })
        else:
            path.append("differences normal -> paired t-test")

        from ..parametric.ttest import paired as paired_ttest
        result = paired_ttest(x1.tolist(), x2.tolist(), alpha=alpha)
        test_name = "Paired t-test"
        test_func = "parametric.ttest.paired"

        for a in result.assumptions:
            assumptions.append({
                "check": a.name, "passed": a.passed,
                "detail": a.detail, "action_taken": a.suggestion or "None needed",
            })
    else:
        path.append(f"differences non-normal, n={n} < 30 -> Wilcoxon signed-rank")
        from ..nonparametric.rank_tests import wilcoxon_signed_rank
        if n < 6:
            path.append("n < 6, falling back to paired t-test")
            from ..parametric.ttest import paired as paired_ttest
            result = paired_ttest(x1.tolist(), x2.tolist(), alpha=alpha)
            test_name = "Paired t-test"
            test_func = "parametric.ttest.paired"
        else:
            result = wilcoxon_signed_rank(x1.tolist(), x2.tolist(), alpha=alpha)
            test_name = "Wilcoxon signed-rank"
            test_func = "nonparametric.rank_tests.wilcoxon_signed_rank"
        assumptions.append({
            "check": "normality_of_differences", "passed": False,
            "detail": "Differences not normal, used non-parametric alternative",
            "action_taken": "Switched to Wilcoxon signed-rank test",
        })

    es = _extract_effect_size(result)
    power_info = _compute_power_paired(x1, x2, alpha)
    interpretation = _interpret_paired(result, names, es, power_info, test_name)
    recs = _recommendations_two_sample(result, prof, power_info)
    confidence = _rate_confidence(prof, power_info, assumptions)

    return AnalysisResult(
        profile=prof,
        test_name=test_name,
        test_function=test_func,
        result=result,
        effect_size=es,
        assumptions=assumptions,
        power=power_info,
        interpretation=interpretation,
        recommendations=recs,
        confidence=confidence,
        decision_path=path,
    )


def _multi_group_independent(
    arrays: list[np.ndarray], names: list[str],
    alpha: float, prof: DataProfile, path: list[str],
) -> AnalysisResult:
    """3+ independent groups: ANOVA or Kruskal-Wallis."""
    all_normal = all(prof.normality.get(n, True) for n in names)
    all_large = all(len(a) >= 30 for a in arrays)
    assumptions = []
    posthoc = None

    if all_normal or all_large:
        if not all_normal:
            path.append("not all groups normal but all n >= 30, using ANOVA (CLT)")
            assumptions.append({
                "check": "normality", "passed": False,
                "detail": "Some groups depart from normality but all samples large",
                "action_taken": "Proceeded with ANOVA due to CLT",
            })
        else:
            path.append("all groups normal -> one-way ANOVA")

        from ..parametric.anova import one_way
        result = one_way(*[a.tolist() for a in arrays], labels=names, alpha=alpha)
        test_name = "One-way ANOVA"
        test_func = "parametric.anova.one_way"

        for a in result.assumptions:
            assumptions.append({
                "check": a.name, "passed": a.passed,
                "detail": a.detail, "action_taken": a.suggestion or "None needed",
            })

        # Auto post-hoc if significant
        if result.significant:
            path.append("ANOVA significant -> running post-hoc")
            if prof.equal_variance:
                path.append("equal variance -> Tukey HSD")
                from ..posthoc.comparisons import tukey_hsd
                posthoc = tukey_hsd(*[a.tolist() for a in arrays], labels=names, alpha=alpha)
            else:
                path.append("unequal variance -> Games-Howell")
                from ..posthoc.comparisons import games_howell
                posthoc = games_howell(*[a.tolist() for a in arrays], labels=names, alpha=alpha)
    else:
        path.append("normality violated -> Kruskal-Wallis H")
        from ..nonparametric.rank_tests import kruskal_wallis
        result = kruskal_wallis(*[a.tolist() for a in arrays], labels=names, alpha=alpha)
        test_name = "Kruskal-Wallis H"
        test_func = "nonparametric.rank_tests.kruskal_wallis"
        assumptions.append({
            "check": "normality", "passed": False,
            "detail": "One or more groups depart from normality",
            "action_taken": "Switched to Kruskal-Wallis (non-parametric)",
        })

        # Auto post-hoc if significant
        if result.significant:
            path.append("Kruskal-Wallis significant -> running Dunn's test")
            from ..posthoc.comparisons import dunn
            posthoc = dunn(*[a.tolist() for a in arrays], labels=names, alpha=alpha)

    es = _extract_effect_size(result)
    power_info = _compute_power_anova(arrays, alpha)
    interpretation = _interpret_multi_group(result, names, es, power_info, test_name, posthoc)
    recs = _recommendations_multi_group(result, prof, power_info, posthoc)
    confidence = _rate_confidence(prof, power_info, assumptions)

    return AnalysisResult(
        profile=prof,
        test_name=test_name,
        test_function=test_func,
        result=result,
        effect_size=es,
        assumptions=assumptions,
        posthoc=posthoc,
        power=power_info,
        interpretation=interpretation,
        recommendations=recs,
        confidence=confidence,
        decision_path=path,
    )


def _multi_group_paired(
    arrays: list[np.ndarray], names: list[str],
    alpha: float, prof: DataProfile, path: list[str],
) -> AnalysisResult:
    """3+ paired/repeated groups: repeated ANOVA or Friedman."""
    assumptions = []
    posthoc = None

    # Check if all groups same length
    lengths = [len(a) for a in arrays]
    if len(set(lengths)) > 1:
        raise ValueError(f"Paired/repeated measures require equal group sizes, got {lengths}")

    all_normal = all(prof.normality.get(n, True) for n in names)

    if all_normal:
        path.append("all conditions normal -> Friedman (repeated measures)")
        # forgestat has Friedman for repeated measures nonparametric
        # and we can use it as a fallback since repeated measures ANOVA isn't always available
    # Use Friedman for repeated measures (robust, always available)
    # Friedman requires 3+ conditions
    path.append("repeated measures -> Friedman test")
    from ..nonparametric.rank_tests import friedman
    result = friedman(*[a.tolist() for a in arrays], labels=names, alpha=alpha)
    test_name = "Friedman"
    test_func = "nonparametric.rank_tests.friedman"

    if not all_normal:
        assumptions.append({
            "check": "normality", "passed": False,
            "detail": "Some conditions depart from normality",
            "action_taken": "Using Friedman test (non-parametric)",
        })

    es = _extract_effect_size(result)
    interpretation = _interpret_multi_group(result, names, es, None, test_name, posthoc)
    recs = _recommendations_multi_group(result, prof, None, posthoc)
    confidence = _rate_confidence(prof, None, assumptions)

    return AnalysisResult(
        profile=prof,
        test_name=test_name,
        test_function=test_func,
        result=result,
        effect_size=es,
        assumptions=assumptions,
        posthoc=posthoc,
        interpretation=interpretation,
        recommendations=recs,
        confidence=confidence,
        decision_path=path,
    )


# ---------------------------------------------------------------------------
# Goal: correlate
# ---------------------------------------------------------------------------

def _analyze_correlate(
    data: Any, alpha: float, labels: list[str] | None,
    path: list[str], **kwargs: Any,
) -> AnalysisResult:
    """Correlation analysis: Pearson or Spearman."""
    # Expect dict of {var_name: values}
    if not isinstance(data, dict):
        raise ValueError("Correlation analysis requires a dict of {variable_name: values}")

    var_names = list(data.keys())
    path.append(f"correlate goal with {len(var_names)} variables")

    # Check normality of each variable to choose method
    arrays = {k: np.asarray(v, dtype=float) for k, v in data.items()}
    all_normal = True
    for name, arr in arrays.items():
        clean = arr[np.isfinite(arr)]
        if len(clean) >= 3:
            chk = check_normality(clean, label=name, alpha=alpha)
            if not chk.passed:
                all_normal = False
                break

    method = kwargs.get("method", None)
    if method is None:
        if all_normal:
            method = "pearson"
            path.append("all variables normal -> Pearson correlation")
        else:
            method = "spearman"
            path.append("non-normality detected -> Spearman correlation")

    from ..parametric.correlation import correlation
    result = correlation(data, method=method, alpha=alpha)

    test_name = f"{method.capitalize()} correlation"
    test_func = "parametric.correlation.correlation"

    prof = DataProfile(
        n_groups=len(var_names),
        group_names=var_names,
        sample_sizes=[len(np.asarray(data[v], dtype=float)) for v in var_names],
        data_type="continuous",
        overall_n=len(np.asarray(data[var_names[0]], dtype=float)),
    )

    # Interpret strongest pair
    es = None
    interpretation = ""
    if result.pairs:
        top = result.pairs[0]
        mag = classify_effect(top.r, "r")
        es = {"name": "r", "value": top.r, "magnitude": mag}
        sig = "significant" if top.p_value < alpha else "not significant"
        direction = "positive" if top.r > 0 else "negative"
        interpretation = (
            f"The {method} correlation between {top.var1} and {top.var2} is "
            f"r = {top.r:.3f} (p = {top.p_value:.4f}), which is {sig}. "
            f"This is a {mag} {direction} association."
        )

    recs = []
    if result.pairs and result.pairs[0].p_value < alpha:
        recs.append("Consider regression analysis to model the relationship")
    if method == "pearson":
        recs.append("Run Spearman correlation to check robustness to non-linearity")

    return AnalysisResult(
        profile=prof,
        test_name=test_name,
        test_function=test_func,
        result=result,
        effect_size=es,
        interpretation=interpretation,
        recommendations=recs,
        confidence="high" if prof.overall_n >= 30 else "moderate",
        decision_path=path,
    )


# ---------------------------------------------------------------------------
# Goal: predict
# ---------------------------------------------------------------------------

def _analyze_predict(
    data: Any, alpha: float, labels: list[str] | None,
    path: list[str], **kwargs: Any,
) -> AnalysisResult:
    """Regression analysis: OLS with diagnostics."""
    if not isinstance(data, dict):
        raise ValueError("Prediction requires a dict with 'y' and 'X' keys, or named variable columns")

    path.append("predict goal -> OLS regression")

    # Accept {y: [...], X: [[...], ...]} or {response: [...], pred1: [...], pred2: [...]}
    if "y" in data and "X" in data:
        y = data["y"]
        X = data["X"]
        feature_names = data.get("feature_names", None)
    else:
        keys = list(data.keys())
        response_key = kwargs.get("response", keys[-1])
        pred_keys = [k for k in keys if k != response_key]
        y = data[response_key]
        X = np.column_stack([np.asarray(data[k], dtype=float) for k in pred_keys])
        feature_names = pred_keys
        path.append(f"response='{response_key}', predictors={pred_keys}")

    from ..regression.linear import ols
    result = ols(X, y, feature_names=feature_names, alpha=alpha)

    n_predictors = len(feature_names) if feature_names else (np.asarray(X).shape[1] if np.asarray(X).ndim == 2 else 1)
    test_name = "Simple OLS" if n_predictors == 1 else "Multiple OLS"
    test_func = "regression.linear.ols"

    es = {"name": "R_squared", "value": result.r_squared, "magnitude": classify_effect(result.r_squared, "r_squared")}

    interpretation = (
        f"The regression model explains {result.r_squared * 100:.1f}% of variance "
        f"(R² = {result.r_squared:.3f}, adj. R² = {result.adj_r_squared:.3f}, "
        f"F = {result.f_statistic:.2f}, p = {result.f_p_value:.4f})."
    )

    recs = []
    if result.r_squared < 0.3:
        recs.append("R² is low — consider additional predictors or non-linear models")
    if result.durbin_watson < 1.5 or result.durbin_watson > 2.5:
        recs.append(f"Durbin-Watson = {result.durbin_watson:.2f} suggests autocorrelation in residuals")

    prof = DataProfile(
        n_groups=1,
        group_names=["regression"],
        sample_sizes=[result.n],
        data_type="continuous",
        overall_n=result.n,
    )

    return AnalysisResult(
        profile=prof,
        test_name=test_name,
        test_function=test_func,
        result=result,
        effect_size=es,
        interpretation=interpretation,
        recommendations=recs,
        confidence="high" if result.n >= 30 else "moderate",
        decision_path=path,
    )


# ---------------------------------------------------------------------------
# Goal: distribute
# ---------------------------------------------------------------------------

def _analyze_distribute(
    data: Any, alpha: float, labels: list[str] | None,
    path: list[str], **kwargs: Any,
) -> AnalysisResult:
    """Distribution analysis: descriptive stats + distribution fitting."""
    groups = _normalize_groups(data, labels)
    name = list(groups.keys())[0]
    x = np.asarray(groups[name], dtype=float)
    x = x[np.isfinite(x)]
    path.append("distribute goal -> descriptive + distribution fit")

    from ..exploratory.univariate import describe
    from ..core.distributions import fit_best

    desc = describe(x.tolist())
    fit = fit_best(x.tolist())

    test_name = "Distribution analysis"
    test_func = "exploratory.univariate.describe + core.distributions.fit_best"

    # Normality check
    norm_check = check_normality(x, label=name, alpha=alpha)

    interpretation = (
        f"The data (n={len(x)}) has mean={desc.mean:.3f}, std={desc.std:.3f}, "
        f"median={desc.median:.3f}. Skewness={desc.skewness:.3f}, "
        f"kurtosis={desc.kurtosis:.3f}. "
        f"Best-fitting distribution: {fit.best.name} "
        f"(KS p={fit.best.ks_p_value:.4f})."
    )
    if norm_check.passed:
        interpretation += " Data is consistent with a normal distribution."
    else:
        interpretation += " Data departs from normality."

    prof = DataProfile(
        n_groups=1,
        group_names=[name],
        sample_sizes=[len(x)],
        data_type="continuous",
        normality={name: norm_check.passed},
        overall_n=len(x),
    )

    return AnalysisResult(
        profile=prof,
        test_name=test_name,
        test_function=test_func,
        result={"descriptive": desc, "fit": fit, "normality": norm_check},
        effect_size=None,
        assumptions=[{
            "check": "normality", "passed": norm_check.passed,
            "detail": norm_check.detail, "action_taken": "Informational",
        }],
        interpretation=interpretation,
        recommendations=["Consider a histogram or Q-Q plot to visualize the distribution"],
        confidence="high" if len(x) >= 30 else "moderate",
        decision_path=path,
    )


# ---------------------------------------------------------------------------
# Goal: equivalence
# ---------------------------------------------------------------------------

def _analyze_equivalence(
    data: Any, alpha: float, margin: float,
    labels: list[str] | None, path: list[str], **kwargs: Any,
) -> AnalysisResult:
    """Equivalence testing: TOST."""
    groups = _normalize_groups(data, labels)
    prof = profile_data(groups, alpha=alpha)
    names = prof.group_names

    if prof.n_groups != 2:
        raise ValueError("Equivalence testing requires exactly 2 groups")

    arrays = [np.asarray(groups[n], dtype=float) for n in names]
    arrays = [a[np.isfinite(a)] for a in arrays]
    path.append(f"equivalence goal, margin={margin} -> TOST")

    from ..parametric.equivalence import tost
    result = tost(arrays[0].tolist(), arrays[1].tolist(), margin=margin, alpha=alpha)
    test_name = "TOST equivalence"
    test_func = "parametric.equivalence.tost"

    assumptions = []
    for a in result.assumptions:
        assumptions.append({
            "check": a.name, "passed": a.passed,
            "detail": a.detail, "action_taken": a.suggestion or "None needed",
        })

    es = None
    if result.effect_size is not None:
        es = {"name": "cohens_d", "value": result.effect_size, "magnitude": classify_effect(result.effect_size, "cohens_d")}

    if result.equivalent:
        interpretation = (
            f"The groups are equivalent within the margin of +/-{margin}. "
            f"Mean difference = {result.mean_diff:.3f}, "
            f"90% CI [{result.ci_lower:.3f}, {result.ci_upper:.3f}] "
            f"falls within the equivalence bounds."
        )
    else:
        interpretation = (
            f"Equivalence not established (TOST p = {result.p_tost:.4f}). "
            f"Mean difference = {result.mean_diff:.3f}, "
            f"90% CI [{result.ci_lower:.3f}, {result.ci_upper:.3f}] "
            f"extends beyond the +/-{margin} equivalence bounds."
        )

    return AnalysisResult(
        profile=prof,
        test_name=test_name,
        test_function=test_func,
        result=result,
        effect_size=es,
        assumptions=assumptions,
        interpretation=interpretation,
        recommendations=[],
        confidence="high" if prof.overall_n >= 60 else "moderate",
        decision_path=path,
    )


# ---------------------------------------------------------------------------
# Helpers: data normalization
# ---------------------------------------------------------------------------

def _normalize_groups(
    data: Any, labels: list[str] | None = None,
) -> dict[str, list[float]]:
    """Normalize various input formats to {name: values} dict."""
    if isinstance(data, dict):
        return {str(k): list(v) for k, v in data.items()}

    if isinstance(data, np.ndarray):
        if data.ndim == 1:
            name = (labels[0] if labels else "Data")
            return {name: data.tolist()}
        elif data.ndim == 2:
            names = labels or [f"Group {i+1}" for i in range(data.shape[1])]
            return {names[i]: data[:, i].tolist() for i in range(data.shape[1])}

    if isinstance(data, list):
        if len(data) == 0:
            return {"Data": []}

        # List of lists/arrays -> multiple groups
        if isinstance(data[0], (list, np.ndarray)):
            names = labels or [f"Group {i+1}" for i in range(len(data))]
            return {names[i]: list(np.asarray(data[i], dtype=float)) for i in range(len(data))}

        # List of numbers -> single group
        if isinstance(data[0], (int, float, np.integer, np.floating)):
            name = (labels[0] if labels else "Data")
            return {name: [float(x) for x in data]}

        # List of dicts -> handled by _rows_to_groups before this point
        raise ValueError("List of dicts requires 'response' and 'factor' parameters")

    raise ValueError(f"Unsupported data type: {type(data)}")


def _rows_to_groups(
    rows: list[dict], response: str, factor: str,
) -> dict[str, list[float]]:
    """Convert list of row dicts to grouped data."""
    groups: dict[str, list[float]] = {}
    for row in rows:
        key = str(row[factor])
        val = float(row[response])
        groups.setdefault(key, []).append(val)
    return groups


def _infer_data_type(arrays: list[np.ndarray]) -> str:
    """Heuristic to infer data type from arrays."""
    all_vals = np.concatenate(arrays) if arrays else np.array([])
    if len(all_vals) == 0:
        return "continuous"

    unique = np.unique(all_vals)

    # Binary
    if len(unique) == 2:
        return "binary"

    # Counts (all non-negative integers)
    if np.all(all_vals >= 0) and np.all(all_vals == np.floor(all_vals)):
        if len(unique) <= 20:
            return "counts"

    # Ordinal (small number of integer-like values)
    if np.all(all_vals == np.floor(all_vals)) and len(unique) <= 10:
        return "ordinal"

    return "continuous"


# ---------------------------------------------------------------------------
# Helpers: effect size extraction
# ---------------------------------------------------------------------------

def _extract_effect_size(result: Any) -> dict | None:
    """Extract effect size info from a test result."""
    if hasattr(result, "effect_size") and result.effect_size is not None:
        es_type = getattr(result, "effect_size_type", "unknown")
        label = getattr(result, "effect_label", classify_effect(result.effect_size, es_type))
        return {"name": es_type, "value": result.effect_size, "magnitude": label}
    return None


# ---------------------------------------------------------------------------
# Helpers: power computation
# ---------------------------------------------------------------------------

def _compute_power_one_sample(x: np.ndarray, mu: float, alpha: float) -> dict | None:
    """Post-hoc power for one-sample test."""
    from ..power.sample_size import power_t_test
    n = len(x)
    d = cohens_d_one_sample(x, mu)
    if abs(d) < 1e-10:
        return {"power": 0.05, "achieved_n": n, "required_n_for_80": None}

    try:
        pw = power_t_test(abs(d), n=n, alpha=alpha, test_type="one_sample")
        req = power_t_test(abs(d), alpha=alpha, power=0.8, test_type="one_sample")
        return {
            "power": pw.power,
            "achieved_n": n,
            "required_n_for_80": req.sample_size,
        }
    except Exception:
        return None


def _compute_power_two_sample(x1: np.ndarray, x2: np.ndarray, alpha: float) -> dict | None:
    """Post-hoc power for two-sample test."""
    from ..power.sample_size import power_t_test
    n = min(len(x1), len(x2))
    d = cohens_d_two_sample(x1, x2)
    if abs(d) < 1e-10:
        return {"power": 0.05, "achieved_n": n, "required_n_for_80": None}

    try:
        pw = power_t_test(abs(d), n=n, alpha=alpha, test_type="two_sample")
        req = power_t_test(abs(d), alpha=alpha, power=0.8, test_type="two_sample")
        return {
            "power": pw.power,
            "achieved_n": n,
            "required_n_for_80": req.sample_size,
        }
    except Exception:
        return None


def _compute_power_paired(x1: np.ndarray, x2: np.ndarray, alpha: float) -> dict | None:
    """Post-hoc power for paired test."""
    from ..power.sample_size import power_t_test
    d = cohens_d_paired(x1, x2)
    n = len(x1)
    if abs(d) < 1e-10 or not math.isfinite(d):
        return {"power": 1.0 if abs(d) == float("inf") else 0.05, "achieved_n": n, "required_n_for_80": None}

    try:
        pw = power_t_test(abs(d), n=n, alpha=alpha, test_type="paired")
        req = power_t_test(abs(d), alpha=alpha, power=0.8, test_type="paired")
        return {
            "power": pw.power,
            "achieved_n": n,
            "required_n_for_80": req.sample_size,
        }
    except Exception:
        return None


def _compute_power_anova(arrays: list[np.ndarray], alpha: float) -> dict | None:
    """Post-hoc power for ANOVA."""
    from ..power.sample_size import power_anova
    from ..core.effect_size import eta_squared as eta_sq_func

    k = len(arrays)
    all_data = np.concatenate(arrays)
    grand_mean = np.mean(all_data)
    ss_between = sum(len(a) * (np.mean(a) - grand_mean) ** 2 for a in arrays)
    ss_total = float(np.sum((all_data - grand_mean) ** 2))
    eta2 = eta_sq_func(ss_between, ss_total)

    if eta2 >= 1.0:
        return None
    f = math.sqrt(eta2 / (1 - eta2)) if eta2 < 1 else 0.0
    n_per = min(len(a) for a in arrays)

    if f < 1e-10:
        return {"power": 0.05, "achieved_n": n_per, "required_n_for_80": None}

    try:
        pw = power_anova(f, k, n_per_group=n_per, alpha=alpha)
        req = power_anova(f, k, alpha=alpha, power=0.8)
        return {
            "power": pw.power,
            "achieved_n": n_per,
            "required_n_for_80": req.sample_size,
        }
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Helpers: interpretation strings
# ---------------------------------------------------------------------------

def _interpret_one_sample(
    result: Any, mu: float, es: dict | None, power_info: dict | None, test_name: str,
) -> str:
    """Generate interpretation for one-sample test."""
    sig = result.significant if hasattr(result, "significant") else result.p_value < 0.05
    p = result.p_value
    stat = result.statistic

    if sig:
        txt = (
            f"The {test_name} found a significant difference from mu={mu} "
            f"(statistic={stat:.3f}, p={p:.4f})."
        )
    else:
        txt = (
            f"The {test_name} found no significant difference from mu={mu} "
            f"(statistic={stat:.3f}, p={p:.4f})."
        )

    if es:
        txt += f" Effect size: {es['name']} = {es['value']:.3f} ({es['magnitude']})."

    if not sig and power_info and power_info.get("power", 1) < 0.8:
        txt += (
            f" Note: achieved power = {power_info['power']:.2f}, "
            f"which is below 0.80. The test may lack power to detect a real difference."
        )

    return txt


def _interpret_two_sample(
    result: Any, names: list[str], es: dict | None, power_info: dict | None, test_name: str,
) -> str:
    """Generate interpretation for two-sample test."""
    sig = result.significant if hasattr(result, "significant") else result.p_value < 0.05
    p = result.p_value

    if sig:
        txt = (
            f"The {test_name} found a significant difference between "
            f"{names[0]} and {names[1]} (p={p:.4f})."
        )
    else:
        txt = (
            f"The {test_name} found no significant difference between "
            f"{names[0]} and {names[1]} (p={p:.4f})."
        )

    if hasattr(result, "ci_lower") and result.ci_lower is not None:
        txt += f" 95% CI for difference: [{result.ci_lower:.3f}, {result.ci_upper:.3f}]."

    if es:
        txt += f" Effect size: {es['name']} = {es['value']:.3f} ({es['magnitude']})."
        if sig and es["magnitude"] in ("negligible", "small"):
            txt += " While statistically significant, the practical effect is small."
        elif sig and es["magnitude"] in ("medium", "large"):
            txt += " This is both statistically and practically significant."

    if not sig and power_info and power_info.get("power", 1) < 0.8:
        txt += (
            f" Note: achieved power = {power_info['power']:.2f}. "
            f"The sample may be too small to detect a real difference."
        )

    return txt


def _interpret_paired(
    result: Any, names: list[str], es: dict | None, power_info: dict | None, test_name: str,
) -> str:
    """Generate interpretation for paired test."""
    return _interpret_two_sample(result, names, es, power_info, test_name)


def _interpret_multi_group(
    result: Any, names: list[str], es: dict | None,
    power_info: dict | None, test_name: str, posthoc: Any | None,
) -> str:
    """Generate interpretation for multi-group test."""
    sig = result.significant if hasattr(result, "significant") else result.p_value < 0.05
    p = result.p_value
    k = len(names)

    if sig:
        txt = (
            f"The {test_name} found a significant difference among the {k} groups "
            f"(p={p:.4f})."
        )
    else:
        txt = (
            f"The {test_name} found no significant difference among the {k} groups "
            f"(p={p:.4f})."
        )

    if es:
        txt += f" Effect size: {es['name']} = {es['value']:.3f} ({es['magnitude']})."

    if posthoc and hasattr(posthoc, "comparisons"):
        sig_pairs = [c for c in posthoc.comparisons if c.significant]
        if sig_pairs:
            pair_strs = [f"{c.group1} vs {c.group2}" for c in sig_pairs]
            txt += f" Post-hoc: significant differences between {', '.join(pair_strs)}."

    if not sig and power_info and power_info.get("power", 1) < 0.8:
        txt += (
            f" Note: achieved power = {power_info['power']:.2f}. "
            f"The sample may be too small to detect real differences."
        )

    return txt


# ---------------------------------------------------------------------------
# Helpers: recommendations
# ---------------------------------------------------------------------------

def _recommendations_one_sample(result: Any, prof: DataProfile, power_info: dict | None) -> list[str]:
    recs = []
    if power_info and power_info.get("power", 1) < 0.8:
        req = power_info.get("required_n_for_80")
        if req:
            recs.append(f"Increase sample size to n={req} for 80% power")
    if any(c > 0 for c in prof.outlier_counts.values()):
        recs.append("Outliers detected — consider robust alternatives or investigate outlier sources")
    return recs


def _recommendations_two_sample(result: Any, prof: DataProfile, power_info: dict | None) -> list[str]:
    recs = []
    if power_info and power_info.get("power", 1) < 0.8:
        req = power_info.get("required_n_for_80")
        if req:
            recs.append(f"Increase sample size to n={req} per group for 80% power")
    if any(c > 0 for c in prof.outlier_counts.values()):
        recs.append("Outliers detected — consider robust alternatives")
    return recs


def _recommendations_multi_group(
    result: Any, prof: DataProfile, power_info: dict | None, posthoc: Any | None,
) -> list[str]:
    recs = []
    sig = result.significant if hasattr(result, "significant") else result.p_value < 0.05
    if sig and posthoc is None:
        recs.append("Run post-hoc comparisons to identify which groups differ")
    if power_info and power_info.get("power", 1) < 0.8:
        req = power_info.get("required_n_for_80")
        if req:
            recs.append(f"Increase sample size to n={req} per group for 80% power")
    if any(c > 0 for c in prof.outlier_counts.values()):
        recs.append("Outliers detected — consider robust alternatives")
    return recs


# ---------------------------------------------------------------------------
# Helpers: confidence rating
# ---------------------------------------------------------------------------

def _rate_confidence(
    prof: DataProfile, power_info: dict | None, assumptions: list[dict],
) -> str:
    """Rate overall confidence in the analysis result."""
    issues = 0

    # Small samples
    if any(s < 5 for s in prof.sample_sizes):
        issues += 2
    elif any(s < 20 for s in prof.sample_sizes):
        issues += 1

    # Low power
    if power_info and power_info.get("power", 1) < 0.5:
        issues += 2
    elif power_info and power_info.get("power", 1) < 0.8:
        issues += 1

    # Assumption violations
    violations = sum(1 for a in assumptions if not a.get("passed", True))
    issues += violations

    if issues == 0:
        return "high"
    elif issues <= 2:
        return "moderate"
    else:
        return "low"
