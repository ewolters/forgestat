"""Distribution fitting and diagnostics.

Fit data to standard distributions, compute goodness-of-fit, identify best fit.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from scipy import stats


@dataclass
class DistributionFit:
    """Result of fitting one distribution to data."""

    name: str
    params: tuple
    ks_statistic: float
    ks_p_value: float
    aic: float = 0.0
    bic: float = 0.0


@dataclass
class FitResult:
    """Result of fitting multiple distributions and selecting best."""

    best: DistributionFit
    all_fits: list[DistributionFit]
    data_summary: dict


# Candidate distributions for automatic fitting
_CANDIDATES = [
    ("normal", stats.norm),
    ("lognormal", stats.lognorm),
    ("weibull", stats.weibull_min),
    ("exponential", stats.expon),
    ("gamma", stats.gamma),
    ("uniform", stats.uniform),
]


def fit_distribution(
    data: list[float] | np.ndarray,
    dist_name: str = "normal",
) -> DistributionFit:
    """Fit a single named distribution to data.

    Args:
        data: Sample values.
        dist_name: Distribution name (normal, lognormal, weibull, exponential, gamma, uniform).

    Returns:
        DistributionFit with parameters and goodness-of-fit.
    """
    x = np.asarray(data, dtype=float)
    x = x[np.isfinite(x)]
    n = len(x)

    dist_map = {d[0]: d[1] for d in _CANDIDATES}
    dist = dist_map.get(dist_name)
    if dist is None:
        raise ValueError(f"Unknown distribution: {dist_name}. Available: {list(dist_map.keys())}")

    params = dist.fit(x)
    # Use the scipy distribution's cdf for KS test
    ks_stat, ks_p = stats.kstest(x, lambda v: dist.cdf(v, *params))

    # AIC/BIC
    log_lik = np.sum(dist.logpdf(x, *params))
    k = len(params)
    aic = 2 * k - 2 * log_lik
    bic = k * math.log(n) - 2 * log_lik if n > 0 else float("inf")

    return DistributionFit(
        name=dist_name,
        params=params,
        ks_statistic=float(ks_stat),
        ks_p_value=float(ks_p),
        aic=float(aic),
        bic=float(bic),
    )


def fit_best(data: list[float] | np.ndarray) -> FitResult:
    """Fit all candidate distributions and select best by AIC.

    Returns:
        FitResult with best fit and all candidates ranked.
    """
    x = np.asarray(data, dtype=float)
    x = x[np.isfinite(x)]

    fits = []
    for name, dist in _CANDIDATES:
        try:
            params = dist.fit(x)
            ks_stat, ks_p = stats.kstest(x, lambda v: dist.cdf(v, *params))
            log_lik = np.sum(dist.logpdf(x, *params))
            k = len(params)
            n = len(x)
            aic = 2 * k - 2 * log_lik
            bic = k * math.log(n) - 2 * log_lik if n > 0 else float("inf")
            fits.append(DistributionFit(
                name=name, params=params,
                ks_statistic=float(ks_stat), ks_p_value=float(ks_p),
                aic=float(aic), bic=float(bic),
            ))
        except Exception:
            continue

    fits.sort(key=lambda f: f.aic)

    summary = {
        "n": len(x),
        "mean": float(np.mean(x)),
        "std": float(np.std(x, ddof=1)) if len(x) > 1 else 0.0,
        "skewness": float(stats.skew(x)) if len(x) > 2 else 0.0,
        "kurtosis": float(stats.kurtosis(x)) if len(x) > 3 else 0.0,
    }

    return FitResult(
        best=fits[0] if fits else DistributionFit("none", (), 0.0, 0.0),
        all_fits=fits,
        data_summary=summary,
    )


def box_cox(data: list[float] | np.ndarray) -> tuple[np.ndarray, float]:
    """Apply Box-Cox transformation.

    Data must be strictly positive. Returns transformed data and lambda.
    """
    x = np.asarray(data, dtype=float)
    x = x[np.isfinite(x)]
    if np.any(x <= 0):
        # Shift to make positive
        shift = abs(np.min(x)) + 1.0
        x = x + shift
    transformed, lmbda = stats.boxcox(x)
    return transformed, float(lmbda)


def johnson_transform(data: list[float] | np.ndarray) -> tuple[np.ndarray, str, tuple]:
    """Apply Johnson transformation to approximate normality.

    Returns (transformed_data, family, params).
    Family is one of: "SU" (unbounded), "SB" (bounded), "SL" (lognormal), "SN" (normal).
    """
    x = np.asarray(data, dtype=float)
    x = x[np.isfinite(x)]

    # Use scipy's Johnson SU as primary
    try:
        params = stats.johnsonsu.fit(x)
        transformed = stats.johnsonsu.ppf(
            stats.johnsonsu.cdf(x, *params),
            0, 1,  # standard normal target
        )
        return transformed, "SU", params
    except Exception:
        # Fallback to Box-Cox
        t, lmbda = box_cox(x)
        return t, "SL", (lmbda,)
