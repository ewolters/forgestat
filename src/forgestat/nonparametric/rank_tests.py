"""Non-parametric rank tests — distribution-free alternatives to parametric tests.

Mann-Whitney U, Kruskal-Wallis, Wilcoxon signed-rank, Friedman, Mood's median, sign test.
"""

from __future__ import annotations

import math

import numpy as np
from scipy import stats

from ..core.effect_size import classify_effect, epsilon_squared, rank_biserial
from ..core.types import RankTestResult, TestResult


def mann_whitney(
    x1: list[float] | np.ndarray,
    x2: list[float] | np.ndarray,
    alpha: float = 0.05,
    alternative: str = "two-sided",
) -> RankTestResult:
    """Mann-Whitney U test: non-parametric alternative to two-sample t-test.

    H₀: The two populations have the same distribution.

    Args:
        x1, x2: Two independent samples.
        alpha: Significance level.
        alternative: "two-sided", "less", or "greater".
    """
    a = np.asarray(x1, dtype=float)
    b = np.asarray(x2, dtype=float)
    a, b = a[np.isfinite(a)], b[np.isfinite(b)]
    n1, n2 = len(a), len(b)

    if n1 < 1 or n2 < 1:
        raise ValueError("Need at least 1 observation per group")

    u_stat, p_val = stats.mannwhitneyu(a, b, alternative=alternative)
    r_rb = rank_biserial(float(u_stat), n1, n2)

    # Hodges-Lehmann median difference
    diffs = np.subtract.outer(a, b).ravel()
    median_diff = float(np.median(diffs))

    return RankTestResult(
        test_name="Mann-Whitney U",
        statistic=float(u_stat),
        p_value=float(p_val),
        effect_size=abs(r_rb),
        effect_size_type="rank_biserial",
        effect_label=classify_effect(r_rb, "rank_biserial"),
        alpha=alpha,
        median1=float(np.median(a)),
        median2=float(np.median(b)),
        median_diff=median_diff,
        n1=n1,
        n2=n2,
        extra={"rank_biserial": r_rb, "hodges_lehmann": median_diff},
    )


def kruskal_wallis(
    *groups: list[float] | np.ndarray,
    labels: list[str] | None = None,
    alpha: float = 0.05,
) -> TestResult:
    """Kruskal-Wallis H test: non-parametric alternative to one-way ANOVA.

    H₀: All k populations have the same distribution.

    Args:
        *groups: Two or more groups.
        labels: Group names.
        alpha: Significance level.
    """
    arrays = [np.asarray(g, dtype=float) for g in groups]
    arrays = [a[np.isfinite(a)] for a in arrays]
    names = labels or [f"Group {i+1}" for i in range(len(arrays))]
    k = len(arrays)

    if k < 2:
        raise ValueError("Need at least 2 groups")

    h_stat, p_val = stats.kruskal(*arrays)
    n_total = sum(len(a) for a in arrays)
    eps2 = epsilon_squared(float(h_stat), n_total)

    group_medians = {name: float(np.median(a)) for name, a in zip(names, arrays)}

    return TestResult(
        test_name="Kruskal-Wallis H",
        statistic=float(h_stat),
        p_value=float(p_val),
        df=float(k - 1),
        effect_size=eps2,
        effect_size_type="epsilon_squared",
        effect_label=classify_effect(eps2, "epsilon_squared"),
        alpha=alpha,
        extra={"group_medians": group_medians, "k": k, "n_total": n_total},
    )


def wilcoxon_signed_rank(
    x1: list[float] | np.ndarray,
    x2: list[float] | np.ndarray | None = None,
    alpha: float = 0.05,
) -> RankTestResult:
    """Wilcoxon signed-rank test: non-parametric alternative to paired t-test.

    If x2 is None, tests x1 against median=0.

    Args:
        x1: First sample (or differences if x2 is None).
        x2: Second sample (paired with x1).
        alpha: Significance level.
    """
    a = np.asarray(x1, dtype=float)

    if x2 is not None:
        b = np.asarray(x2, dtype=float)
        if len(a) != len(b):
            raise ValueError("Paired samples must have equal length")
        mask = np.isfinite(a) & np.isfinite(b)
        diff = a[mask] - b[mask]
    else:
        diff = a[np.isfinite(a)]

    n = len(diff)
    if n < 6:
        raise ValueError(f"Need at least 6 observations for Wilcoxon test, got {n}")

    # Remove zeros (tied with hypothesized median)
    diff_nz = diff[diff != 0]
    n_eff = len(diff_nz)

    if n_eff < 1:
        return RankTestResult(
            test_name="Wilcoxon signed-rank",
            statistic=0.0, p_value=1.0, alpha=alpha,
            median1=float(np.median(diff)), n1=n,
        )

    w_stat, p_val = stats.wilcoxon(diff_nz)

    # Effect size: r = Z / sqrt(N)
    z = stats.norm.ppf(1 - p_val / 2) if p_val < 1 else 0.0
    r = z / math.sqrt(n_eff) if n_eff > 0 else 0.0

    return RankTestResult(
        test_name="Wilcoxon signed-rank",
        statistic=float(w_stat),
        p_value=float(p_val),
        effect_size=abs(r),
        effect_size_type="r",
        effect_label=classify_effect(r, "r"),
        alpha=alpha,
        median1=float(np.median(diff)),
        n1=n,
        extra={"z": float(z), "n_effective": n_eff},
    )


def friedman(
    *groups: list[float] | np.ndarray,
    labels: list[str] | None = None,
    alpha: float = 0.05,
) -> TestResult:
    """Friedman test: non-parametric alternative to repeated measures ANOVA.

    H₀: All k treatments have the same effect.

    Args:
        *groups: k repeated measurements (same subjects, same length).
        labels: Condition names.
        alpha: Significance level.
    """
    arrays = [np.asarray(g, dtype=float) for g in groups]
    k = len(arrays)
    n = len(arrays[0])

    if k < 3:
        raise ValueError("Need at least 3 conditions for Friedman test")
    if any(len(a) != n for a in arrays):
        raise ValueError("All conditions must have equal number of observations")

    chi2, p_val = stats.friedmanchisquare(*arrays)
    names = labels or [f"Condition {i+1}" for i in range(k)]

    # Kendall's W effect size
    w = float(chi2) / (n * (k - 1)) if n > 0 and k > 1 else 0.0

    return TestResult(
        test_name="Friedman",
        statistic=float(chi2),
        p_value=float(p_val),
        df=float(k - 1),
        effect_size=w,
        effect_size_type="kendalls_w",
        effect_label=classify_effect(w, "r"),
        alpha=alpha,
        extra={
            "kendalls_w": w,
            "condition_medians": {name: float(np.median(a)) for name, a in zip(names, arrays)},
            "k": k,
            "n": n,
        },
    )


def mood_median(
    *groups: list[float] | np.ndarray,
    labels: list[str] | None = None,
    alpha: float = 0.05,
) -> TestResult:
    """Mood's median test: robust test for equality of group medians.

    More robust to outliers than Kruskal-Wallis.

    Args:
        *groups: Two or more groups.
        labels: Group names.
        alpha: Significance level.
    """
    arrays = [np.asarray(g, dtype=float) for g in groups]
    arrays = [a[np.isfinite(a)] for a in arrays]
    names = labels or [f"Group {i+1}" for i in range(len(arrays))]

    if len(arrays) < 2:
        raise ValueError("Need at least 2 groups")

    stat, p_val, grand_median, contingency = stats.median_test(*arrays)

    group_medians = {name: float(np.median(a)) for name, a in zip(names, arrays)}

    return TestResult(
        test_name="Mood's median",
        statistic=float(stat),
        p_value=float(p_val),
        df=float(len(arrays) - 1),
        alpha=alpha,
        extra={
            "grand_median": float(grand_median),
            "group_medians": group_medians,
            "contingency_table": contingency.tolist(),
        },
    )


def sign_test(
    data: list[float] | np.ndarray,
    median0: float = 0.0,
    alpha: float = 0.05,
) -> TestResult:
    """Sign test: simplest non-parametric test for median.

    H₀: population median = median0.
    Uses only the direction (sign) of deviations, not magnitudes.

    Args:
        data: Sample values.
        median0: Hypothesized median.
        alpha: Significance level.
    """
    x = np.asarray(data, dtype=float)
    x = x[np.isfinite(x)]

    above = int(np.sum(x > median0))
    below = int(np.sum(x < median0))
    ties = int(np.sum(x == median0))
    n_eff = above + below

    if n_eff == 0:
        return TestResult(
            test_name="Sign test", statistic=0.0, p_value=1.0, alpha=alpha,
        )

    # Two-sided binomial test
    p_val = float(stats.binomtest(above, n_eff, 0.5).pvalue)

    return TestResult(
        test_name="Sign test",
        statistic=float(above),
        p_value=p_val,
        alpha=alpha,
        significant=bool(p_val < alpha),
        extra={
            "above": above,
            "below": below,
            "ties": ties,
            "n_effective": n_eff,
            "sample_median": float(np.median(x)),
            "hypothesized_median": median0,
        },
    )


def runs_test(
    data: list[float] | np.ndarray,
    cutoff: float | None = None,
    alpha: float = 0.05,
) -> TestResult:
    """Wald-Wolfowitz runs test for randomness.

    Tests whether the sequence of values above/below a cutoff is random.
    H₀: The sequence is random.

    Args:
        data: Sequence of values.
        cutoff: Threshold (default: median). Values above = "+", below = "-".
        alpha: Significance level.

    Returns:
        TestResult — significant means non-random (clustering or alternation).
    """
    x = np.asarray(data, dtype=float)
    x = x[np.isfinite(x)]
    n = len(x)

    if n < 10:
        raise ValueError(f"Need at least 10 observations for runs test, got {n}")

    if cutoff is None:
        cutoff = float(np.median(x))

    # Classify above/below
    signs = x > cutoff
    n_plus = int(np.sum(signs))
    n_minus = n - n_plus

    if n_plus == 0 or n_minus == 0:
        return TestResult(
            test_name="Runs test", statistic=0.0, p_value=1.0, alpha=alpha,
            extra={"n_plus": n_plus, "n_minus": n_minus, "n_runs": 0},
        )

    # Count runs
    runs = 1
    for i in range(1, n):
        if signs[i] != signs[i - 1]:
            runs += 1

    # Expected runs and variance under H0
    mu_r = 1 + 2 * n_plus * n_minus / n
    if n > 1:
        var_r = (2 * n_plus * n_minus * (2 * n_plus * n_minus - n)) / (n ** 2 * (n - 1))
    else:
        var_r = 0

    if var_r > 0:
        z = (runs - mu_r) / math.sqrt(var_r)
        p_val = 2 * (1 - stats.norm.cdf(abs(z)))
    else:
        z = 0.0
        p_val = 1.0

    return TestResult(
        test_name="Runs test",
        statistic=float(z),
        p_value=float(p_val),
        alpha=alpha,
        extra={
            "n_runs": runs,
            "expected_runs": float(mu_r),
            "n_plus": n_plus,
            "n_minus": n_minus,
        },
    )
