"""Variance tests — F-test, Bartlett's, Levene's.

Wrappers around scipy for consistent API.
"""

from __future__ import annotations


import numpy as np
from scipy import stats

from ..core.types import TestResult


def f_test(
    x1: list[float] | np.ndarray,
    x2: list[float] | np.ndarray,
    alpha: float = 0.05,
) -> TestResult:
    """F-test for equality of two variances: H₀: σ₁² = σ₂².

    Args:
        x1, x2: Two samples.
        alpha: Significance level.
    """
    a = np.asarray(x1, dtype=float)
    b = np.asarray(x2, dtype=float)
    a, b = a[np.isfinite(a)], b[np.isfinite(b)]

    var1 = float(np.var(a, ddof=1))
    var2 = float(np.var(b, ddof=1))
    n1, n2 = len(a), len(b)

    f_stat = var1 / var2 if var2 > 0 else float("inf")
    df1, df2 = n1 - 1, n2 - 1
    p_val = 2 * min(stats.f.cdf(f_stat, df1, df2), 1 - stats.f.cdf(f_stat, df1, df2))

    return TestResult(
        test_name="F-test for variances",
        statistic=float(f_stat),
        p_value=float(p_val),
        df=float(df1),
        alpha=alpha,
        extra={"var1": var1, "var2": var2, "df1": df1, "df2": df2, "ratio": f_stat},
    )


def variance_test(
    *groups: list[float] | np.ndarray,
    labels: list[str] | None = None,
    alpha: float = 0.05,
    method: str = "levene",
) -> TestResult:
    """Test equality of variances across groups.

    Args:
        *groups: Two or more samples.
        labels: Group names.
        alpha: Significance level.
        method: "levene" (robust) or "bartlett" (assumes normality).
    """
    arrays = [np.asarray(g, dtype=float) for g in groups]
    arrays = [a[np.isfinite(a)] for a in arrays]

    if method == "bartlett":
        stat, p = stats.bartlett(*arrays)
        name = "Bartlett's test"
    else:
        stat, p = stats.levene(*arrays, center="median")
        name = "Levene's test"

    return TestResult(
        test_name=name,
        statistic=float(stat),
        p_value=float(p),
        df=float(len(arrays) - 1),
        alpha=alpha,
        extra={"variances": [float(np.var(a, ddof=1)) for a in arrays]},
    )
