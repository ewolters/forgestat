"""Chi-square tests — independence and goodness-of-fit.

Pure computation. Returns ChiSquareResult dataclass.
"""

from __future__ import annotations

import numpy as np
from scipy import stats

from ..core.effect_size import classify_effect, cramers_v
from ..core.types import ChiSquareResult


def chi_square_independence(
    observed: list[list[float]] | np.ndarray,
    row_labels: list[str] | None = None,
    col_labels: list[str] | None = None,
    alpha: float = 0.05,
) -> ChiSquareResult:
    """Chi-square test of independence on a contingency table.

    Args:
        observed: 2D array/list of observed counts.
        row_labels: Names for rows.
        col_labels: Names for columns.
        alpha: Significance level.

    Returns:
        ChiSquareResult with χ², p-value, Cramér's V.
    """
    obs = np.asarray(observed, dtype=float)
    r, c = obs.shape

    chi2, p, dof, expected = stats.chi2_contingency(obs)
    n = float(np.sum(obs))
    min_dim = min(r, c)
    v = cramers_v(float(chi2), int(n), min_dim)

    return ChiSquareResult(
        test_name="Chi-square test of independence",
        statistic=float(chi2),
        p_value=float(p),
        df=float(dof),
        effect_size=v,
        effect_size_type="cramers_v",
        effect_label=classify_effect(v, "cramers_v"),
        alpha=alpha,
        observed=obs.tolist(),
        expected=expected.tolist(),
        row_labels=row_labels or [f"Row {i+1}" for i in range(r)],
        col_labels=col_labels or [f"Col {j+1}" for j in range(c)],
        cramers_v=v,
    )


def chi_square_goodness_of_fit(
    observed: list[float] | np.ndarray,
    expected: list[float] | np.ndarray | None = None,
    categories: list[str] | None = None,
    alpha: float = 0.05,
) -> ChiSquareResult:
    """Chi-square goodness-of-fit test.

    Args:
        observed: Observed counts per category.
        expected: Expected counts (default: uniform).
        categories: Category labels.
        alpha: Significance level.
    """
    obs = np.asarray(observed, dtype=float)
    k = len(obs)

    if expected is None:
        exp = np.full(k, np.sum(obs) / k)
    else:
        exp = np.asarray(expected, dtype=float)

    chi2, p = stats.chisquare(obs, f_exp=exp)
    dof = k - 1
    n = float(np.sum(obs))
    v = cramers_v(float(chi2), int(n), k) if k > 1 else 0.0

    cats = categories or [f"Cat {i+1}" for i in range(k)]

    return ChiSquareResult(
        test_name="Chi-square goodness-of-fit",
        statistic=float(chi2),
        p_value=float(p),
        df=float(dof),
        effect_size=v,
        effect_size_type="cramers_v",
        effect_label=classify_effect(v, "cramers_v"),
        alpha=alpha,
        observed=[obs.tolist()],
        expected=[exp.tolist()],
        row_labels=["Observed"],
        col_labels=cats,
        cramers_v=v,
    )


def fisher_exact(
    table: list[list[int]] | np.ndarray,
    alpha: float = 0.05,
) -> ChiSquareResult:
    """Fisher's exact test for 2×2 contingency tables.

    Args:
        table: 2×2 contingency table.
        alpha: Significance level.
    """
    obs = np.asarray(table, dtype=float)
    if obs.shape != (2, 2):
        raise ValueError(f"Fisher's exact test requires a 2×2 table, got {obs.shape}")

    odds_ratio, p = stats.fisher_exact(obs.astype(int))

    return ChiSquareResult(
        test_name="Fisher's exact test",
        statistic=float(odds_ratio),
        p_value=float(p),
        df=None,
        alpha=alpha,
        observed=obs.tolist(),
        expected=[],
        cramers_v=0.0,
        extra={"odds_ratio": float(odds_ratio)},
    )
