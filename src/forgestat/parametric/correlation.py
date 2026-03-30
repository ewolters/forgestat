"""Correlation analysis — Pearson, Spearman, Kendall.

Pure computation. Returns CorrelationResult dataclass.
"""

from __future__ import annotations

import math

import numpy as np
from scipy import stats

from ..core.types import CorrelationPair, CorrelationResult


def correlation(
    data: dict[str, list[float]],
    method: str = "pearson",
    alpha: float = 0.05,
) -> CorrelationResult:
    """Compute pairwise correlations for all variable pairs.

    Args:
        data: Dict of variable_name → values (all same length).
        method: "pearson", "spearman", or "kendall".
        alpha: Significance level.

    Returns:
        CorrelationResult with matrix and pairwise details.
    """
    names = list(data.keys())
    arrays = {k: np.asarray(v, dtype=float) for k, v in data.items()}
    k = len(names)

    if k < 2:
        raise ValueError("Need at least 2 variables for correlation")

    corr_func = {
        "pearson": stats.pearsonr,
        "spearman": stats.spearmanr,
        "kendall": stats.kendalltau,
    }.get(method)

    if corr_func is None:
        raise ValueError(f"Unknown method: {method}. Use 'pearson', 'spearman', or 'kendall'.")

    matrix: dict[str, dict[str, float]] = {n: {} for n in names}
    pairs: list[CorrelationPair] = []

    for i in range(k):
        for j in range(k):
            xi = arrays[names[i]]
            xj = arrays[names[j]]

            # Pairwise complete
            mask = np.isfinite(xi) & np.isfinite(xj)
            xi_clean = xi[mask]
            xj_clean = xj[mask]
            n = len(xi_clean)

            if n < 3:
                matrix[names[i]][names[j]] = float("nan")
                continue

            r, p = corr_func(xi_clean, xj_clean)
            r = float(r)
            p = float(p)
            matrix[names[i]][names[j]] = r

            if i < j:
                # Fisher z CI for Pearson/Spearman
                ci_lo, ci_hi = _fisher_z_ci(r, n, 1 - alpha)

                pairs.append(CorrelationPair(
                    var1=names[i],
                    var2=names[j],
                    r=r,
                    p_value=p,
                    n=n,
                    ci_lower=ci_lo,
                    ci_upper=ci_hi,
                    r_squared=r ** 2,
                ))

    # Sort by absolute r descending
    pairs.sort(key=lambda p: abs(p.r), reverse=True)

    return CorrelationResult(
        method=method,
        pairs=pairs,
        matrix=matrix,
    )


def _fisher_z_ci(r: float, n: int, conf: float = 0.95) -> tuple[float, float]:
    """Fisher z-transformation CI for a correlation coefficient."""
    if n < 4 or abs(r) >= 1.0:
        return (r, r)

    z_r = 0.5 * math.log((1 + r) / (1 - r))
    se = 1.0 / math.sqrt(n - 3)
    z_crit = stats.norm.ppf((1 + conf) / 2)

    z_lo = z_r - z_crit * se
    z_hi = z_r + z_crit * se

    r_lo = math.tanh(z_lo)
    r_hi = math.tanh(z_hi)

    return (float(r_lo), float(r_hi))
