"""Effect size computation and classification.

Cohen's d, η², ω², Cramér's V, r, rank-biserial — with magnitude labels.
"""

from __future__ import annotations

import math

import numpy as np


# Thresholds from Cohen (1988) and common conventions
_THRESHOLDS = {
    "cohens_d": [(0.2, "negligible"), (0.5, "small"), (0.8, "medium"), (float("inf"), "large")],
    "eta_squared": [(0.01, "negligible"), (0.06, "small"), (0.14, "medium"), (float("inf"), "large")],
    "omega_squared": [(0.01, "negligible"), (0.06, "small"), (0.14, "medium"), (float("inf"), "large")],
    "cramers_v": [(0.1, "negligible"), (0.3, "small"), (0.5, "medium"), (float("inf"), "large")],
    "r": [(0.1, "negligible"), (0.3, "small"), (0.5, "medium"), (float("inf"), "large")],
    "r_squared": [(0.01, "negligible"), (0.09, "small"), (0.25, "medium"), (float("inf"), "large")],
    "rank_biserial": [(0.1, "negligible"), (0.3, "small"), (0.5, "medium"), (float("inf"), "large")],
    "epsilon_squared": [(0.01, "negligible"), (0.06, "small"), (0.14, "medium"), (float("inf"), "large")],
}


def classify_effect(value: float, effect_type: str) -> str:
    """Classify effect size magnitude.

    Args:
        value: Absolute effect size value.
        effect_type: One of the keys in _THRESHOLDS.

    Returns:
        "negligible", "small", "medium", or "large"
    """
    thresholds = _THRESHOLDS.get(effect_type, _THRESHOLDS["cohens_d"])
    v = abs(value)
    for cutoff, label in thresholds:
        if v < cutoff:
            return label
    return "large"


def cohens_d_one_sample(data: list[float] | np.ndarray, mu: float = 0.0) -> float:
    """Cohen's d for one-sample test: (mean - mu) / s."""
    x = np.asarray(data, dtype=float)
    x = x[np.isfinite(x)]
    if len(x) < 2:
        return 0.0
    return float((np.mean(x) - mu) / np.std(x, ddof=1))


def cohens_d_two_sample(
    x1: list[float] | np.ndarray,
    x2: list[float] | np.ndarray,
    pooled: bool = True,
) -> float:
    """Cohen's d for two independent samples.

    Args:
        pooled: If True, use pooled SD. If False, use Welch-like SD.
    """
    a = np.asarray(x1, dtype=float)
    b = np.asarray(x2, dtype=float)
    a, b = a[np.isfinite(a)], b[np.isfinite(b)]
    n1, n2 = len(a), len(b)

    if n1 < 2 or n2 < 2:
        return 0.0

    if pooled:
        sp = math.sqrt(((n1 - 1) * np.var(a, ddof=1) + (n2 - 1) * np.var(b, ddof=1)) / (n1 + n2 - 2))
    else:
        sp = math.sqrt((np.var(a, ddof=1) + np.var(b, ddof=1)) / 2)

    if sp == 0:
        return 0.0
    return float((np.mean(a) - np.mean(b)) / sp)


def cohens_d_paired(
    x1: list[float] | np.ndarray,
    x2: list[float] | np.ndarray,
) -> float:
    """Cohen's d for paired samples: mean(diff) / sd(diff).

    When all differences are identical (sd=0), returns inf (with sign)
    if mean_diff != 0, else 0.
    """
    a = np.asarray(x1, dtype=float)
    b = np.asarray(x2, dtype=float)
    diff = a - b
    diff = diff[np.isfinite(diff)]

    if len(diff) < 2:
        return 0.0
    mean_diff = float(np.mean(diff))
    sd = float(np.std(diff, ddof=1))
    if sd == 0:
        return float("inf") * (1 if mean_diff > 0 else -1) if mean_diff != 0 else 0.0
    return mean_diff / sd


def eta_squared(ss_between: float, ss_total: float) -> float:
    """η² = SS_between / SS_total."""
    if ss_total <= 0:
        return 0.0
    return ss_between / ss_total


def omega_squared(ss_between: float, ss_total: float, df_between: float, ms_within: float) -> float:
    """ω² — less biased estimate of variance explained.

    ω² = (SS_between - df_between * MS_within) / (SS_total + MS_within)
    """
    if ss_total + ms_within <= 0:
        return 0.0
    return max(0.0, (ss_between - df_between * ms_within) / (ss_total + ms_within))


def partial_eta_squared(ss_effect: float, ss_error: float) -> float:
    """Partial η² = SS_effect / (SS_effect + SS_error)."""
    total = ss_effect + ss_error
    if total <= 0:
        return 0.0
    return ss_effect / total


def cramers_v(chi2_stat: float, n: int, min_dim: int) -> float:
    """Cramér's V = sqrt(χ² / (n * (min(r,c) - 1)))."""
    denom = n * max(min_dim - 1, 1)
    if denom <= 0:
        return 0.0
    return math.sqrt(chi2_stat / denom)


def rank_biserial(u_stat: float, n1: int, n2: int) -> float:
    """Rank-biserial correlation from Mann-Whitney U.

    r = 1 - 2U / (n1 * n2)
    """
    product = n1 * n2
    if product == 0:
        return 0.0
    return 1.0 - 2.0 * u_stat / product


def epsilon_squared(h_stat: float, n: int) -> float:
    """ε² effect size for Kruskal-Wallis H test.

    ε² = (H - k + 1) / (n - k), simplified as H / (n² - 1) / (n - 1)
    Using simpler form: H / (n - 1).
    """
    if n <= 1:
        return 0.0
    return h_stat / (n * n - 1)
