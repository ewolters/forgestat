"""Equivalence testing — TOST (Two One-Sided Tests).

Pure computation. Returns EquivalenceResult dataclass.
"""

from __future__ import annotations

import math

import numpy as np
from scipy import stats

from ..core.assumptions import check_normality, check_outliers
from ..core.effect_size import cohens_d_two_sample
from ..core.types import EquivalenceResult


def tost(
    x1: list[float] | np.ndarray,
    x2: list[float] | np.ndarray,
    margin: float,
    alpha: float = 0.05,
) -> EquivalenceResult:
    """Two One-Sided Tests (TOST) for equivalence.

    H₀: |μ₁ - μ₂| ≥ margin  vs  H₁: |μ₁ - μ₂| < margin

    Args:
        x1, x2: Two independent samples.
        margin: Equivalence margin (symmetric ±margin).
        alpha: Significance level (note: TOST uses α, CI uses 1-2α).
    """
    a = np.asarray(x1, dtype=float)
    b = np.asarray(x2, dtype=float)
    a, b = a[np.isfinite(a)], b[np.isfinite(b)]
    n1, n2 = len(a), len(b)

    if n1 < 2 or n2 < 2:
        raise ValueError("Need at least 2 observations per group")

    if margin <= 0:
        raise ValueError("Equivalence margin must be positive")

    diff = float(np.mean(a) - np.mean(b))

    # Welch SE and df
    s1 = float(np.var(a, ddof=1))
    s2 = float(np.var(b, ddof=1))
    se = math.sqrt(s1 / n1 + s2 / n2)

    if se > 0:
        num = (s1 / n1 + s2 / n2) ** 2
        den = (s1 / n1) ** 2 / (n1 - 1) + (s2 / n2) ** 2 / (n2 - 1)
        df = num / den if den > 0 else n1 + n2 - 2
    else:
        df = n1 + n2 - 2

    # Two one-sided tests
    t_lower = (diff - (-margin)) / se if se > 0 else 0.0
    t_upper = (diff - margin) / se if se > 0 else 0.0

    p_lower = float(1 - stats.t.cdf(t_lower, df))
    p_upper = float(stats.t.cdf(t_upper, df))
    p_tost = max(p_lower, p_upper)

    equivalent = p_tost < alpha

    # 90% CI (1-2α for TOST) and 95% CI
    ci_level = 1 - 2 * alpha
    t_crit = stats.t.ppf(1 - alpha, df)
    ci_lo = diff - t_crit * se
    ci_hi = diff + t_crit * se

    d = cohens_d_two_sample(a, b)

    assumptions = [
        check_normality(a, label="Group 1"),
        check_normality(b, label="Group 2"),
        check_outliers(a, label="Group 1"),
        check_outliers(b, label="Group 2"),
    ]

    return EquivalenceResult(
        mean_diff=diff,
        margin=margin,
        t_lower=float(t_lower),
        t_upper=float(t_upper),
        p_lower=p_lower,
        p_upper=p_upper,
        p_tost=p_tost,
        equivalent=equivalent,
        ci_lower=float(ci_lo),
        ci_upper=float(ci_hi),
        ci_level=ci_level,
        effect_size=d,
        assumptions=assumptions,
    )
