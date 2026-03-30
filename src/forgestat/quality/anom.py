"""Analysis of Means (ANOM) — compare group means to overall mean.

Quality-specific alternative to ANOVA. Tests whether any group mean
falls outside decision limits (similar logic to control chart limits).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
from scipy import stats


@dataclass
class ANOMGroup:
    """One group in ANOM analysis."""

    name: str
    mean: float
    n: int
    exceeds_upper: bool = False
    exceeds_lower: bool = False


@dataclass
class ANOMResult:
    """Analysis of Means result."""

    grand_mean: float = 0.0
    upper_limit: float = 0.0
    lower_limit: float = 0.0
    mse: float = 0.0
    h_critical: float = 0.0
    groups: list[ANOMGroup] = field(default_factory=list)
    any_significant: bool = False
    alpha: float = 0.05
    n_groups: int = 0


def anom(
    *groups: list[float] | np.ndarray,
    labels: list[str] | None = None,
    alpha: float = 0.05,
) -> ANOMResult:
    """Analysis of Means — compare each group mean to the grand mean.

    Decision limits based on the critical value from the
    Studentized Maximum Modulus distribution (approximated via t).

    Args:
        *groups: Two or more groups.
        labels: Group names.
        alpha: Significance level.

    Returns:
        ANOMResult with decision limits and flagged groups.
    """
    arrays = [np.asarray(g, dtype=float) for g in groups]
    arrays = [a[np.isfinite(a)] for a in arrays]
    names = labels or [f"Group {i+1}" for i in range(len(arrays))]
    k = len(arrays)
    ns = [len(a) for a in arrays]
    N = sum(ns)

    if k < 2:
        raise ValueError("Need at least 2 groups")

    grand_mean = float(np.mean(np.concatenate(arrays)))

    # MSE (within-group variance)
    ss_within = sum(float(np.sum((a - np.mean(a)) ** 2)) for a in arrays)
    df_within = N - k
    mse = ss_within / df_within if df_within > 0 else 0

    # Critical value approximation
    # Exact ANOM uses Studentized Maximum Modulus; approximate via Bonferroni-corrected t
    t_crit = stats.t.ppf(1 - alpha / (2 * k), df_within) if df_within > 0 else 0

    result_groups = []
    any_sig = False

    for name, arr, n_i in zip(names, arrays, ns):
        group_mean = float(np.mean(arr))

        # Decision limits for this group size
        h = t_crit * math.sqrt(mse * (1 - n_i / N) / n_i) if n_i > 0 and N > 0 and mse > 0 else 0
        upper = grand_mean + h
        lower = grand_mean - h

        exceeds_upper = group_mean > upper
        exceeds_lower = group_mean < lower

        if exceeds_upper or exceeds_lower:
            any_sig = True

        result_groups.append(ANOMGroup(
            name=name,
            mean=group_mean,
            n=n_i,
            exceeds_upper=exceeds_upper,
            exceeds_lower=exceeds_lower,
        ))

    # Use the first group's limits as representative (for balanced designs they're all the same)
    n_rep = ns[0] if ns else 1
    h_rep = t_crit * math.sqrt(mse * (1 - n_rep / N) / n_rep) if n_rep > 0 and N > 0 and mse > 0 else 0

    return ANOMResult(
        grand_mean=grand_mean,
        upper_limit=grand_mean + h_rep,
        lower_limit=grand_mean - h_rep,
        mse=mse,
        h_critical=h_rep,
        groups=result_groups,
        any_significant=any_sig,
        alpha=alpha,
        n_groups=k,
    )
