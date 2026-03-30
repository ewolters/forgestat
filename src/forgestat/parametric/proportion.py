"""Proportion tests — one-sample and two-sample z-tests for proportions.

Pure computation. Returns ProportionResult dataclass.
"""

from __future__ import annotations

import math

from scipy import stats

from ..core.effect_size import classify_effect
from ..core.types import ProportionResult


def one_proportion(
    successes: int,
    n: int,
    p0: float = 0.5,
    alpha: float = 0.05,
    conf: float = 0.95,
) -> ProportionResult:
    """One-sample z-test for proportion: H₀: p = p0.

    Args:
        successes: Number of successes.
        n: Total observations.
        p0: Hypothesized proportion.
        alpha: Significance level.
        conf: Confidence level.
    """
    if n <= 0:
        raise ValueError("n must be positive")

    p_hat = successes / n
    se = math.sqrt(p0 * (1 - p0) / n)
    z = (p_hat - p0) / se if se > 0 else 0.0
    p_val = 2 * (1 - stats.norm.cdf(abs(z)))

    # Wilson CI
    z_crit = stats.norm.ppf((1 + conf) / 2)
    denom = 1 + z_crit ** 2 / n
    center = (p_hat + z_crit ** 2 / (2 * n)) / denom
    margin = z_crit * math.sqrt((p_hat * (1 - p_hat) + z_crit ** 2 / (4 * n)) / n) / denom
    ci_lo = max(0, center - margin)
    ci_hi = min(1, center + margin)

    # Cohen's h effect size
    h = 2 * (math.asin(math.sqrt(p_hat)) - math.asin(math.sqrt(p0)))

    return ProportionResult(
        test_name="One-proportion z-test",
        statistic=z,
        p_value=p_val,
        ci_lower=ci_lo,
        ci_upper=ci_hi,
        effect_size=abs(h),
        effect_size_type="cohens_h",
        effect_label=classify_effect(h, "cohens_d"),  # same thresholds
        alpha=alpha,
        p_hat=p_hat,
        n1=n,
    )


def two_proportions(
    successes1: int,
    n1: int,
    successes2: int,
    n2: int,
    alpha: float = 0.05,
    conf: float = 0.95,
) -> ProportionResult:
    """Two-sample z-test for proportions: H₀: p₁ = p₂.

    Args:
        successes1, n1: Successes and total for group 1.
        successes2, n2: Successes and total for group 2.
    """
    if n1 <= 0 or n2 <= 0:
        raise ValueError("Both n1 and n2 must be positive")

    p1 = successes1 / n1
    p2 = successes2 / n2
    p_diff = p1 - p2

    # Pooled proportion under H0
    p_pool = (successes1 + successes2) / (n1 + n2)
    se = math.sqrt(p_pool * (1 - p_pool) * (1 / n1 + 1 / n2))
    z = p_diff / se if se > 0 else 0.0
    p_val = 2 * (1 - stats.norm.cdf(abs(z)))

    # Unpooled CI for difference
    se_diff = math.sqrt(p1 * (1 - p1) / n1 + p2 * (1 - p2) / n2)
    z_crit = stats.norm.ppf((1 + conf) / 2)
    ci_lo = p_diff - z_crit * se_diff
    ci_hi = p_diff + z_crit * se_diff

    h = 2 * (math.asin(math.sqrt(p1)) - math.asin(math.sqrt(p2)))

    return ProportionResult(
        test_name="Two-proportion z-test",
        statistic=z,
        p_value=p_val,
        ci_lower=ci_lo,
        ci_upper=ci_hi,
        effect_size=abs(h),
        effect_size_type="cohens_h",
        effect_label=classify_effect(h, "cohens_d"),
        alpha=alpha,
        p_hat=p1,
        p_hat2=p2,
        p_diff=p_diff,
        n1=n1,
        n2=n2,
    )
