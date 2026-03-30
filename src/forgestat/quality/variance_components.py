"""Variance components analysis — nested designs.

Decomposes total variation into between-group and within-group components.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class VarianceComponent:
    """One source in variance decomposition."""

    source: str
    variance: float = 0.0
    pct_contribution: float = 0.0
    std_dev: float = 0.0


@dataclass
class VarianceComponentsResult:
    """Variance components analysis result."""

    components: list[VarianceComponent] = field(default_factory=list)
    total_variance: float = 0.0
    icc: float = 0.0  # intraclass correlation


def one_way_random(
    groups: dict[str, list[float]],
    factor_name: str = "Factor",
) -> VarianceComponentsResult:
    """One-way random effects variance components.

    Decomposes into between-group (σ²_between) and within-group (σ²_within).
    ICC = σ²_between / (σ²_between + σ²_within).

    Args:
        groups: Dict of group_name → values.
        factor_name: Name for the between-group source.

    Returns:
        VarianceComponentsResult with components and ICC.
    """
    arrays = [np.asarray(v, dtype=float) for v in groups.values()]
    k = len(arrays)
    ns = [len(a) for a in arrays]
    N = sum(ns)

    grand_mean = np.mean(np.concatenate(arrays))

    # ANOVA SS
    ss_between = sum(n * (np.mean(a) - grand_mean) ** 2 for n, a in zip(ns, arrays))
    ss_within = sum(np.sum((a - np.mean(a)) ** 2) for a in arrays)

    df_between = k - 1
    df_within = N - k

    ms_between = ss_between / df_between if df_between > 0 else 0
    ms_within = ss_within / df_within if df_within > 0 else 0

    # n₀ for unbalanced design
    n0 = (N - sum(n ** 2 for n in ns) / N) / (k - 1) if k > 1 else 1

    # Variance components (ANOVA method)
    sigma2_within = ms_within
    sigma2_between = max(0, (ms_between - ms_within) / n0)
    sigma2_total = sigma2_between + sigma2_within

    icc = sigma2_between / sigma2_total if sigma2_total > 0 else 0.0

    components = [
        VarianceComponent(
            source=factor_name,
            variance=float(sigma2_between),
            pct_contribution=100 * sigma2_between / sigma2_total if sigma2_total > 0 else 0,
            std_dev=float(np.sqrt(sigma2_between)),
        ),
        VarianceComponent(
            source="Within (Error)",
            variance=float(sigma2_within),
            pct_contribution=100 * sigma2_within / sigma2_total if sigma2_total > 0 else 0,
            std_dev=float(np.sqrt(sigma2_within)),
        ),
    ]

    return VarianceComponentsResult(
        components=components,
        total_variance=float(sigma2_total),
        icc=float(icc),
    )
