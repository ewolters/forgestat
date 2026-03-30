"""Repeated measures ANOVA — within-subjects designs.

Sphericity testing (Mauchly), Greenhouse-Geisser and Huynh-Feldt corrections.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
from scipy import stats


@dataclass
class RepeatedMeasuresResult:
    """Repeated measures ANOVA result."""

    f_statistic: float = 0.0
    p_value: float = 0.0
    p_value_gg: float = 0.0  # Greenhouse-Geisser corrected
    df_condition: float = 0.0
    df_error: float = 0.0
    ss_condition: float = 0.0
    ss_subject: float = 0.0
    ss_error: float = 0.0
    ms_condition: float = 0.0
    ms_error: float = 0.0
    partial_eta_sq: float = 0.0
    epsilon_gg: float = 0.0  # Greenhouse-Geisser epsilon
    sphericity_met: bool = True
    mauchly_p: float | None = None
    condition_means: dict[str, float] = field(default_factory=dict)
    n_subjects: int = 0
    n_conditions: int = 0


def repeated_measures_anova(
    data: dict[str, list[float]],
    condition_names: list[str] | None = None,
) -> RepeatedMeasuresResult:
    """One-way repeated measures ANOVA.

    Each key in data is a condition, each value is the measurements
    for all subjects in that condition (same length, same order).

    Args:
        data: {condition_name: [subject1_score, subject2_score, ...]}.
        condition_names: Optional explicit names.

    Returns:
        RepeatedMeasuresResult with F, p, sphericity, corrections.
    """
    names = condition_names or list(data.keys())
    k = len(names)
    arrays = [np.asarray(data[n], dtype=float) for n in names]
    n = len(arrays[0])

    if any(len(a) != n for a in arrays):
        raise ValueError("All conditions must have the same number of subjects")
    if k < 2:
        raise ValueError("Need at least 2 conditions")
    if n < 2:
        raise ValueError("Need at least 2 subjects")

    # Stack into n × k matrix
    Y = np.column_stack(arrays)
    grand_mean = float(np.mean(Y))

    # Means
    cond_means = np.mean(Y, axis=0)  # k condition means
    subj_means = np.mean(Y, axis=1)  # n subject means

    # Sum of squares
    ss_total = float(np.sum((Y - grand_mean) ** 2))
    ss_condition = n * float(np.sum((cond_means - grand_mean) ** 2))
    ss_subject = k * float(np.sum((subj_means - grand_mean) ** 2))
    ss_error = ss_total - ss_condition - ss_subject

    df_condition = k - 1
    df_subject = n - 1
    df_error = df_condition * df_subject

    ms_condition = ss_condition / df_condition if df_condition > 0 else 0
    ms_error = ss_error / df_error if df_error > 0 else 0

    if ms_error > 1e-15:
        f_stat = ms_condition / ms_error
    elif ms_condition > 0:
        f_stat = 1e6  # perfect separation — zero error SS
    else:
        f_stat = 0.0
    p_val = float(1 - stats.f.cdf(f_stat, df_condition, df_error)) if df_error > 0 else 1.0

    # Partial eta²
    partial_eta = ss_condition / (ss_condition + ss_error) if (ss_condition + ss_error) > 0 else 0

    # Greenhouse-Geisser epsilon
    # Based on covariance matrix of differences
    epsilon_gg = _greenhouse_geisser_epsilon(Y, k)

    # Corrected p-value
    df1_gg = df_condition * epsilon_gg
    df2_gg = df_error * epsilon_gg
    p_val_gg = float(1 - stats.f.cdf(f_stat, max(1, df1_gg), max(1, df2_gg)))

    # Mauchly's test (approximate)
    mauchly_p = _mauchly_test(Y, k, n)
    sphericity_met = mauchly_p is None or mauchly_p > 0.05

    return RepeatedMeasuresResult(
        f_statistic=float(f_stat),
        p_value=float(p_val),
        p_value_gg=p_val_gg,
        df_condition=float(df_condition),
        df_error=float(df_error),
        ss_condition=float(ss_condition),
        ss_subject=float(ss_subject),
        ss_error=float(ss_error),
        ms_condition=float(ms_condition),
        ms_error=float(ms_error),
        partial_eta_sq=float(partial_eta),
        epsilon_gg=float(epsilon_gg),
        sphericity_met=sphericity_met,
        mauchly_p=mauchly_p,
        condition_means={name: float(m) for name, m in zip(names, cond_means)},
        n_subjects=n,
        n_conditions=k,
    )


def _greenhouse_geisser_epsilon(Y: np.ndarray, k: int) -> float:
    """Compute Greenhouse-Geisser epsilon for sphericity correction."""
    # Compute covariance of differences
    # Center each condition
    Y_centered = Y - np.mean(Y, axis=1, keepdims=True)
    S = np.cov(Y_centered.T)  # k × k covariance matrix

    if S.ndim == 0:
        return 1.0

    # Epsilon = (trace(S))² / ((k-1) * trace(S @ S))
    tr_S = np.trace(S)
    tr_S2 = np.trace(S @ S)

    if tr_S2 == 0 or k <= 1:
        return 1.0

    epsilon = tr_S ** 2 / ((k - 1) * tr_S2)
    # Bound: 1/(k-1) ≤ ε ≤ 1
    return max(1 / (k - 1), min(1.0, float(epsilon)))


def _mauchly_test(Y: np.ndarray, k: int, n: int) -> float | None:
    """Approximate Mauchly's test of sphericity.

    Returns p-value. If p < 0.05, sphericity is violated.
    """
    if k <= 2:
        return None  # sphericity is always met with 2 conditions

    # Contrast matrix (orthonormalized differences)
    C = np.zeros((k, k - 1))
    for j in range(k - 1):
        C[j, j] = 1
        C[j + 1, j] = -1
    C = C / np.sqrt(2)

    # Transformed data
    Z = Y @ C
    S = np.cov(Z.T)

    if S.ndim == 0:
        return None

    p = k - 1  # number of contrasts
    det_S = np.linalg.det(S)
    tr_S = np.trace(S)

    if tr_S == 0 or p == 0:
        return None

    # Mauchly's W
    W = det_S / (tr_S / p) ** p

    if W <= 0 or W > 1:
        return None

    # Chi-square approximation
    df = p * (p + 1) / 2 - 1
    f = 1 - (2 * p ** 2 + p + 2) / (6 * p * (n - 1))
    chi2 = -f * (n - 1) * math.log(W)

    p_val = float(1 - stats.chi2.cdf(chi2, df)) if df > 0 else None
    return p_val
