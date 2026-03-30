"""Univariate descriptive profiling and diagnostics.

Summary statistics, shape analysis, outlier detection, bootstrap CI.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import stats


@dataclass
class DescriptiveStats:
    """Comprehensive univariate summary."""

    n: int = 0
    n_missing: int = 0
    mean: float = 0.0
    std: float = 0.0
    se_mean: float = 0.0
    variance: float = 0.0
    median: float = 0.0
    mode: float | None = None
    q1: float = 0.0
    q3: float = 0.0
    iqr: float = 0.0
    min: float = 0.0
    max: float = 0.0
    range: float = 0.0
    skewness: float = 0.0
    kurtosis: float = 0.0  # excess kurtosis
    cv: float = 0.0  # coefficient of variation
    n_outliers: int = 0
    shape_description: str = ""


@dataclass
class BootstrapCI:
    """Bootstrap confidence interval result."""

    statistic: str
    estimate: float
    ci_lower: float
    ci_upper: float
    ci_level: float
    n_bootstrap: int
    method: str = "percentile"


@dataclass
class ToleranceInterval:
    """Tolerance interval result."""

    lower: float
    upper: float
    coverage: float
    confidence: float
    k_factor: float
    method: str = "normal"  # "normal" or "nonparametric"


def describe(
    data: list[float] | np.ndarray,
) -> DescriptiveStats:
    """Comprehensive univariate descriptive statistics.

    Args:
        data: Sample values (NaN/inf allowed — will be counted and excluded).

    Returns:
        DescriptiveStats with all common summary measures.
    """
    raw = np.asarray(data, dtype=float)
    n_missing = int(np.sum(~np.isfinite(raw)))
    x = raw[np.isfinite(raw)]
    n = len(x)

    if n == 0:
        return DescriptiveStats(n=0, n_missing=n_missing)

    mean = float(np.mean(x))
    std = float(np.std(x, ddof=1)) if n > 1 else 0.0
    var = std ** 2
    se = std / np.sqrt(n) if n > 0 else 0.0
    median = float(np.median(x))
    q1, q3 = float(np.percentile(x, 25)), float(np.percentile(x, 75))
    iqr = q3 - q1
    mn, mx = float(np.min(x)), float(np.max(x))

    skew = float(stats.skew(x)) if n > 2 else 0.0
    kurt = float(stats.kurtosis(x)) if n > 3 else 0.0  # excess
    cv = std / abs(mean) * 100 if abs(mean) > 1e-15 else 0.0

    # Outliers (IQR)
    lower_fence = q1 - 1.5 * iqr
    upper_fence = q3 + 1.5 * iqr
    n_outliers = int(np.sum((x < lower_fence) | (x > upper_fence)))

    # Shape description
    parts = []
    if abs(skew) < 0.5:
        parts.append("approximately symmetric")
    elif skew > 0:
        parts.append("right-skewed")
    else:
        parts.append("left-skewed")

    if kurt > 1:
        parts.append("heavy-tailed")
    elif kurt < -1:
        parts.append("light-tailed")

    return DescriptiveStats(
        n=n,
        n_missing=n_missing,
        mean=mean,
        std=std,
        se_mean=float(se),
        variance=var,
        median=median,
        q1=q1,
        q3=q3,
        iqr=iqr,
        min=mn,
        max=mx,
        range=mx - mn,
        skewness=skew,
        kurtosis=kurt,
        cv=cv,
        n_outliers=n_outliers,
        shape_description=", ".join(parts),
    )


def bootstrap_ci(
    data: list[float] | np.ndarray,
    statistic: str = "mean",
    ci_level: float = 0.95,
    n_bootstrap: int = 10000,
    seed: int = 42,
) -> BootstrapCI:
    """Bootstrap confidence interval for a statistic.

    Args:
        data: Sample values.
        statistic: "mean", "median", "std", or "trimmed_mean".
        ci_level: Confidence level.
        n_bootstrap: Number of bootstrap resamples.
        seed: RNG seed for reproducibility.

    Returns:
        BootstrapCI with estimate and percentile interval.
    """
    x = np.asarray(data, dtype=float)
    x = x[np.isfinite(x)]
    n = len(x)
    rng = np.random.default_rng(seed)

    stat_func = {
        "mean": np.mean,
        "median": np.median,
        "std": lambda a: float(np.std(a, ddof=1)),
        "trimmed_mean": lambda a: float(stats.trim_mean(a, 0.1)),
    }.get(statistic, np.mean)

    estimate = float(stat_func(x))

    # Bootstrap
    boot_stats = np.empty(n_bootstrap)
    for i in range(n_bootstrap):
        sample = rng.choice(x, size=n, replace=True)
        boot_stats[i] = stat_func(sample)

    alpha = (1 - ci_level) / 2
    ci_lo = float(np.percentile(boot_stats, 100 * alpha))
    ci_hi = float(np.percentile(boot_stats, 100 * (1 - alpha)))

    return BootstrapCI(
        statistic=statistic,
        estimate=estimate,
        ci_lower=ci_lo,
        ci_upper=ci_hi,
        ci_level=ci_level,
        n_bootstrap=n_bootstrap,
    )


def tolerance_interval(
    data: list[float] | np.ndarray,
    coverage: float = 0.95,
    confidence: float = 0.95,
    method: str = "normal",
) -> ToleranceInterval:
    """Compute a tolerance interval.

    Normal method: x̄ ± k·s where k depends on n, coverage, confidence.
    Nonparametric: uses order statistics.

    Args:
        data: Sample values.
        coverage: Proportion of population to cover.
        confidence: Confidence level.
        method: "normal" or "nonparametric".

    Returns:
        ToleranceInterval with bounds and k-factor.
    """
    x = np.asarray(data, dtype=float)
    x = x[np.isfinite(x)]
    n = len(x)

    if n < 2:
        raise ValueError("Need at least 2 observations")

    mean = float(np.mean(x))
    s = float(np.std(x, ddof=1))

    if method == "normal":
        # k = z_p * sqrt((n-1) / chi2(1-gamma, n-1)) * sqrt(1 + 1/n)
        # Simplified one-sided, then convert to two-sided
        z_p = stats.norm.ppf((1 + coverage) / 2)
        chi2_val = stats.chi2.ppf(1 - confidence, n - 1)
        k = z_p * np.sqrt((n - 1) / chi2_val) * np.sqrt(1 + 1 / n)
        k = float(k)

        return ToleranceInterval(
            lower=mean - k * s,
            upper=mean + k * s,
            coverage=coverage,
            confidence=confidence,
            k_factor=k,
            method="normal",
        )
    else:
        # Nonparametric: use order statistics
        # For 95/95, need at least 59 observations for the min/max to work
        x_sorted = np.sort(x)
        # Simple approach: use percentiles
        lo_pct = (1 - coverage) / 2 * 100
        hi_pct = (1 - (1 - coverage) / 2) * 100

        return ToleranceInterval(
            lower=float(np.percentile(x_sorted, lo_pct)),
            upper=float(np.percentile(x_sorted, hi_pct)),
            coverage=coverage,
            confidence=confidence,
            k_factor=0.0,
            method="nonparametric",
        )
