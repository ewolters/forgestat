"""Assumption checking — normality, equal variance, outliers.

Pure computation. Returns AssumptionCheck objects.
"""

from __future__ import annotations

import numpy as np
from scipy import stats

from .types import AssumptionCheck


def check_normality(
    data: list[float] | np.ndarray,
    label: str = "Data",
    alpha: float = 0.05,
    method: str = "auto",
) -> AssumptionCheck:
    """Test normality using Shapiro-Wilk (n<=5000) or Anderson-Darling.

    Args:
        data: Sample data.
        label: Name for reporting.
        alpha: Significance level.
        method: "shapiro", "anderson", "ks", or "auto" (Shapiro if n<=5000).

    Returns:
        AssumptionCheck with pass/fail and details.
    """
    x = np.asarray(data, dtype=float)
    x = x[np.isfinite(x)]
    n = len(x)

    if n < 3:
        return AssumptionCheck(
            name="normality",
            test_name="insufficient_data",
            passed=True,
            detail=f"{label}: n={n}, too few observations to test normality",
        )

    if method == "auto":
        method = "shapiro" if n <= 5000 else "anderson"

    if method == "shapiro":
        stat, p = stats.shapiro(x)
        return AssumptionCheck(
            name="normality",
            test_name="Shapiro-Wilk",
            statistic=float(stat),
            p_value=float(p),
            passed=bool(p >= alpha),
            detail=f"{label}: W={stat:.4f}, p={p:.4f}",
            suggestion="" if p >= alpha else f"{label} departs from normality — consider non-parametric alternative",
        )

    elif method == "anderson":
        result = stats.anderson(x, dist="norm")
        # Use 5% critical value (index 2 in anderson result)
        cv_idx = min(2, len(result.critical_values) - 1)
        cv = result.critical_values[cv_idx]
        passed = bool(result.statistic < cv)
        return AssumptionCheck(
            name="normality",
            test_name="Anderson-Darling",
            statistic=float(result.statistic),
            passed=passed,
            detail=f"{label}: A²={result.statistic:.4f}, critical={cv:.4f} at 5%",
            suggestion="" if passed else f"{label} departs from normality",
        )

    else:  # ks
        stat, p = stats.kstest(x, "norm", args=(np.mean(x), np.std(x, ddof=1)))
        return AssumptionCheck(
            name="normality",
            test_name="Kolmogorov-Smirnov",
            statistic=float(stat),
            p_value=float(p),
            passed=bool(p >= alpha),
            detail=f"{label}: D={stat:.4f}, p={p:.4f}",
            suggestion="" if p >= alpha else f"{label} departs from normality",
        )


def check_equal_variance(
    *groups: list[float] | np.ndarray,
    labels: list[str] | None = None,
    alpha: float = 0.05,
) -> AssumptionCheck:
    """Test equality of variances using Levene's test (median-based).

    Args:
        *groups: Two or more samples.
        labels: Group names for reporting.
        alpha: Significance level.

    Returns:
        AssumptionCheck with pass/fail.
    """
    arrays = [np.asarray(g, dtype=float) for g in groups]
    arrays = [a[np.isfinite(a)] for a in arrays]

    if len(arrays) < 2:
        return AssumptionCheck(
            name="equal_variance",
            test_name="Levene",
            passed=True,
            detail="Fewer than 2 groups — variance test not applicable",
        )

    if any(len(a) < 2 for a in arrays):
        return AssumptionCheck(
            name="equal_variance",
            test_name="Levene",
            passed=True,
            detail="Group(s) have fewer than 2 observations",
        )

    stat, p = stats.levene(*arrays, center="median")
    variances = [float(np.var(a, ddof=1)) for a in arrays]
    ratio = max(variances) / min(variances) if min(variances) > 0 else float("inf")

    names = labels or [f"Group {i+1}" for i in range(len(arrays))]
    var_str = ", ".join(f"{n}: s²={v:.4f}" for n, v in zip(names, variances))

    return AssumptionCheck(
        name="equal_variance",
        test_name="Levene",
        statistic=float(stat),
        p_value=float(p),
        passed=bool(p >= alpha),
        detail=f"W={stat:.4f}, p={p:.4f}, ratio={ratio:.2f}. {var_str}",
        suggestion="" if p >= alpha else "Unequal variances — use Welch's correction or non-parametric test",
    )


def check_outliers(
    data: list[float] | np.ndarray,
    label: str = "Data",
    method: str = "iqr",
    threshold: float = 1.5,
) -> AssumptionCheck:
    """Detect outliers using IQR or Z-score method.

    Args:
        data: Sample data.
        label: Name for reporting.
        method: "iqr" (default) or "zscore".
        threshold: IQR multiplier (default 1.5) or Z threshold (default 3.0).

    Returns:
        AssumptionCheck with outlier count in detail.
    """
    x = np.asarray(data, dtype=float)
    x = x[np.isfinite(x)]

    if len(x) < 4:
        return AssumptionCheck(
            name="outliers",
            test_name=method,
            passed=True,
            detail=f"{label}: too few observations for outlier detection",
        )

    if method == "zscore":
        z = np.abs(stats.zscore(x))
        n_outliers = int(np.sum(z > (threshold if threshold > 1.5 else 3.0)))
    else:  # iqr
        q1, q3 = np.percentile(x, [25, 75])
        iqr = q3 - q1
        lower = q1 - threshold * iqr
        upper = q3 + threshold * iqr
        n_outliers = int(np.sum((x < lower) | (x > upper)))

    pct = 100 * n_outliers / len(x)
    passed = pct < 10  # flag if >10% outliers

    return AssumptionCheck(
        name="outliers",
        test_name=method.upper(),
        passed=passed,
        detail=f"{label}: {n_outliers}/{len(x)} outliers ({pct:.1f}%)",
        suggestion="" if passed else f"{label} has >10% outliers — results may be unreliable",
    )
