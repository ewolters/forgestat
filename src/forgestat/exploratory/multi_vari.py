"""Multi-vari analysis — stratified variance decomposition.

Visualizes and quantifies variation across nested factors.
Common in manufacturing: part × operator × within-part variation.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class VariSource:
    """One factor's contribution to total variation."""

    factor: str
    variance: float = 0.0
    pct_contribution: float = 0.0
    n_levels: int = 0
    level_means: dict[str, float] = field(default_factory=dict)


@dataclass
class MultiVariResult:
    """Multi-vari analysis result."""

    sources: list[VariSource] = field(default_factory=list)
    within_variance: float = 0.0
    within_pct: float = 0.0
    total_variance: float = 0.0
    dominant_source: str = ""
    grand_mean: float = 0.0


def multi_vari(
    data: dict[str, list],
    response: str,
    factors: list[str],
) -> MultiVariResult:
    """Multi-vari analysis — decompose variation by factors.

    For each factor, computes between-level variance as a fraction of total.
    Identifies the dominant source of variation.

    Args:
        data: Dict of column_name → list of values.
        response: Response variable name.
        factors: Factor column names (in nesting order, outermost first).

    Returns:
        MultiVariResult with per-factor variance contributions.
    """
    y = np.asarray(data[response], dtype=float)
    n = len(y)
    grand_mean = float(np.mean(y))
    total_var = float(np.var(y, ddof=1)) if n > 1 else 0.0

    sources = []
    explained_var = 0.0

    for factor in factors:
        f_vals = np.asarray(data[factor])
        levels = sorted(set(f_vals))
        n_levels = len(levels)

        level_means = {}
        between_ss = 0.0
        for lev in levels:
            mask = f_vals == lev
            lev_mean = float(np.mean(y[mask]))
            level_means[str(lev)] = lev_mean
            between_ss += np.sum(mask) * (lev_mean - grand_mean) ** 2

        between_var = between_ss / (n - 1) if n > 1 else 0.0
        pct = 100 * between_var / total_var if total_var > 0 else 0.0
        explained_var += between_var

        sources.append(VariSource(
            factor=factor,
            variance=float(between_var),
            pct_contribution=float(pct),
            n_levels=n_levels,
            level_means=level_means,
        ))

    within_var = max(0, total_var - explained_var)
    within_pct = 100 * within_var / total_var if total_var > 0 else 0.0

    dominant = max(sources, key=lambda s: s.pct_contribution).factor if sources else ""

    return MultiVariResult(
        sources=sources,
        within_variance=float(within_var),
        within_pct=float(within_pct),
        total_variance=float(total_var),
        dominant_source=dominant,
        grand_mean=grand_mean,
    )
