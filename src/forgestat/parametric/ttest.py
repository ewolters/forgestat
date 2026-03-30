"""T-tests — one-sample, two-sample (Welch/pooled), paired.

Pure computation. Returns TTestResult dataclasses.
"""

from __future__ import annotations

import math

import numpy as np
from scipy import stats

from ..core.assumptions import check_normality, check_outliers
from ..core.effect_size import (
    classify_effect,
    cohens_d_one_sample,
    cohens_d_paired,
    cohens_d_two_sample,
)
from ..core.types import TTestResult


def one_sample(
    data: list[float] | np.ndarray,
    mu: float = 0.0,
    alpha: float = 0.05,
    conf: float = 0.95,
) -> TTestResult:
    """One-sample t-test: H₀: μ = mu.

    Args:
        data: Sample values.
        mu: Hypothesized population mean.
        alpha: Significance level.
        conf: Confidence level for interval.

    Returns:
        TTestResult with t-statistic, p-value, CI, Cohen's d.
    """
    x = np.asarray(data, dtype=float)
    x = x[np.isfinite(x)]
    n = len(x)

    if n < 2:
        raise ValueError("Need at least 2 observations for t-test")

    mean = float(np.mean(x))
    se = float(np.std(x, ddof=1) / math.sqrt(n))
    t_stat, p_val = stats.ttest_1samp(x, mu)
    df = n - 1

    ci = stats.t.interval(conf, df, loc=mean, scale=se)
    d = cohens_d_one_sample(x, mu)

    assumptions = [
        check_normality(x, label="Sample"),
        check_outliers(x, label="Sample"),
    ]

    return TTestResult(
        test_name="One-sample t-test",
        statistic=float(t_stat),
        p_value=float(p_val),
        df=float(df),
        ci_lower=float(ci[0]),
        ci_upper=float(ci[1]),
        effect_size=d,
        effect_size_type="cohens_d",
        effect_label=classify_effect(d, "cohens_d"),
        alpha=alpha,
        mean1=mean,
        mean_diff=mean - mu,
        se=se,
        n1=n,
        method="one_sample",
        assumptions=assumptions,
    )


def two_sample(
    x1: list[float] | np.ndarray,
    x2: list[float] | np.ndarray,
    equal_var: bool = False,
    alpha: float = 0.05,
    conf: float = 0.95,
) -> TTestResult:
    """Two-sample t-test: H₀: μ₁ = μ₂.

    Default is Welch's t-test (equal_var=False).

    Args:
        x1, x2: Two independent samples.
        equal_var: If True, use pooled variance (Student's t).
        alpha: Significance level.
        conf: Confidence level for interval.
    """
    a = np.asarray(x1, dtype=float)
    b = np.asarray(x2, dtype=float)
    a, b = a[np.isfinite(a)], b[np.isfinite(b)]
    n1, n2 = len(a), len(b)

    if n1 < 2 or n2 < 2:
        raise ValueError("Need at least 2 observations per group")

    mean1 = float(np.mean(a))
    mean2 = float(np.mean(b))
    diff = mean1 - mean2

    t_stat, p_val = stats.ttest_ind(a, b, equal_var=equal_var)

    # Degrees of freedom
    if equal_var:
        df = n1 + n2 - 2
        sp2 = ((n1 - 1) * np.var(a, ddof=1) + (n2 - 1) * np.var(b, ddof=1)) / df
        se = float(math.sqrt(sp2 * (1 / n1 + 1 / n2)))
    else:
        # Welch-Satterthwaite
        s1 = float(np.var(a, ddof=1))
        s2 = float(np.var(b, ddof=1))
        se = float(math.sqrt(s1 / n1 + s2 / n2))
        if se > 0:
            num = (s1 / n1 + s2 / n2) ** 2
            den = (s1 / n1) ** 2 / (n1 - 1) + (s2 / n2) ** 2 / (n2 - 1)
            df = num / den if den > 0 else n1 + n2 - 2
        else:
            df = n1 + n2 - 2

    ci = stats.t.interval(conf, df, loc=diff, scale=se) if se > 0 else (diff, diff)
    d = cohens_d_two_sample(a, b, pooled=equal_var)

    from ..core.assumptions import check_equal_variance
    assumptions = [
        check_normality(a, label="Group 1"),
        check_normality(b, label="Group 2"),
        check_equal_variance(a, b, labels=["Group 1", "Group 2"]),
        check_outliers(a, label="Group 1"),
        check_outliers(b, label="Group 2"),
    ]

    return TTestResult(
        test_name="Welch's t-test" if not equal_var else "Student's t-test",
        statistic=float(t_stat),
        p_value=float(p_val),
        df=float(df),
        ci_lower=float(ci[0]),
        ci_upper=float(ci[1]),
        effect_size=d,
        effect_size_type="cohens_d",
        effect_label=classify_effect(d, "cohens_d"),
        alpha=alpha,
        mean1=mean1,
        mean2=mean2,
        mean_diff=diff,
        se=se,
        n1=n1,
        n2=n2,
        method="welch" if not equal_var else "pooled",
        assumptions=assumptions,
    )


def paired(
    x1: list[float] | np.ndarray,
    x2: list[float] | np.ndarray,
    alpha: float = 0.05,
    conf: float = 0.95,
) -> TTestResult:
    """Paired t-test: H₀: μ_d = 0.

    Args:
        x1, x2: Paired observations (same length).
        alpha: Significance level.
        conf: Confidence level.
    """
    a = np.asarray(x1, dtype=float)
    b = np.asarray(x2, dtype=float)

    if len(a) != len(b):
        raise ValueError(f"Paired samples must have equal length: {len(a)} vs {len(b)}")

    mask = np.isfinite(a) & np.isfinite(b)
    a, b = a[mask], b[mask]
    n = len(a)

    if n < 2:
        raise ValueError("Need at least 2 paired observations")

    diff = a - b
    mean_diff = float(np.mean(diff))
    sd_diff = float(np.std(diff, ddof=1))
    se = sd_diff / math.sqrt(n)

    t_stat, p_val = stats.ttest_rel(a, b)
    df = n - 1

    ci = stats.t.interval(conf, df, loc=mean_diff, scale=se) if se > 0 else (mean_diff, mean_diff)
    d = cohens_d_paired(a, b)

    assumptions = [
        check_normality(diff, label="Differences"),
        check_outliers(diff, label="Differences"),
    ]

    return TTestResult(
        test_name="Paired t-test",
        statistic=float(t_stat),
        p_value=float(p_val),
        df=float(df),
        ci_lower=float(ci[0]),
        ci_upper=float(ci[1]),
        effect_size=d,
        effect_size_type="cohens_d",
        effect_label=classify_effect(d, "cohens_d"),
        alpha=alpha,
        mean1=float(np.mean(a)),
        mean2=float(np.mean(b)),
        mean_diff=mean_diff,
        se=se,
        n1=n,
        method="paired",
        assumptions=assumptions,
    )
