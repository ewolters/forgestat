"""ANOVA — one-way, two-way, repeated measures.

Pure computation. Returns AnovaResult / Anova2Result dataclasses.
"""

from __future__ import annotations

import numpy as np
from scipy import stats

from ..core.assumptions import check_equal_variance, check_normality
from ..core.effect_size import (
    classify_effect,
    eta_squared,
    omega_squared,
    partial_eta_squared,
)
from ..core.types import Anova2Result, Anova2Source, AnovaResult


def one_way(
    *groups: list[float] | np.ndarray,
    labels: list[str] | None = None,
    alpha: float = 0.05,
) -> AnovaResult:
    """One-way ANOVA: H₀: μ₁ = μ₂ = ... = μ_k.

    Args:
        *groups: Two or more groups of observations.
        labels: Group names.
        alpha: Significance level.
    """
    arrays = [np.asarray(g, dtype=float) for g in groups]
    arrays = [a[np.isfinite(a)] for a in arrays]
    names = labels or [f"Group {i+1}" for i in range(len(arrays))]
    k = len(arrays)

    if k < 2:
        raise ValueError("Need at least 2 groups for ANOVA")

    f_stat, p_val = stats.f_oneway(*arrays)

    # Manual SS decomposition
    all_data = np.concatenate(arrays)
    grand_mean = float(np.mean(all_data))
    n_total = len(all_data)

    ss_between = sum(len(a) * (np.mean(a) - grand_mean) ** 2 for a in arrays)
    ss_total = float(np.sum((all_data - grand_mean) ** 2))
    ss_within = ss_total - ss_between

    df_between = k - 1
    df_within = n_total - k
    ms_between = ss_between / df_between if df_between > 0 else 0
    ms_within = ss_within / df_within if df_within > 0 else 0

    eta2 = eta_squared(ss_between, ss_total)
    omega2 = omega_squared(ss_between, ss_total, df_between, ms_within)

    group_means = {name: float(np.mean(a)) for name, a in zip(names, arrays)}
    group_ns = {name: len(a) for name, a in zip(names, arrays)}

    assumptions = [
        check_equal_variance(*arrays, labels=names),
    ]
    for name, a in zip(names, arrays):
        assumptions.append(check_normality(a, label=name))

    return AnovaResult(
        test_name="One-way ANOVA",
        statistic=float(f_stat),
        p_value=float(p_val),
        df=float(df_between),
        effect_size=eta2,
        effect_size_type="eta_squared",
        effect_label=classify_effect(eta2, "eta_squared"),
        alpha=alpha,
        df_between=float(df_between),
        df_within=float(df_within),
        ss_between=float(ss_between),
        ss_within=float(ss_within),
        ss_total=float(ss_total),
        ms_between=float(ms_between),
        ms_within=float(ms_within),
        omega_squared=omega2,
        group_means=group_means,
        group_ns=group_ns,
        assumptions=assumptions,
    )


def one_way_from_dict(
    groups: dict[str, list[float]],
    alpha: float = 0.05,
) -> AnovaResult:
    """One-way ANOVA from a dict of {group_name: values}.

    Convenience wrapper around one_way().
    """
    names = list(groups.keys())
    arrays = list(groups.values())
    return one_way(*arrays, labels=names, alpha=alpha)


def two_way(
    data: dict[str, list],
    response: str,
    factor_a: str,
    factor_b: str,
    alpha: float = 0.05,
) -> Anova2Result:
    """Two-way ANOVA with interaction.

    Args:
        data: Dict of column_name → list of values (equal length).
        response: Name of response variable column.
        factor_a: Name of first factor column.
        factor_b: Name of second factor column.
        alpha: Significance level.

    Returns:
        Anova2Result with source table (A, B, A:B, Residual).
    """
    y = np.asarray(data[response], dtype=float)
    a_vals = np.asarray(data[factor_a])
    b_vals = np.asarray(data[factor_b])
    n = len(y)

    grand_mean = float(np.mean(y))

    # Unique levels
    a_levels = sorted(set(a_vals))
    b_levels = sorted(set(b_vals))

    # Cell means
    cell_means = {}
    a_means = {}
    b_means = {}

    for al in a_levels:
        mask_a = a_vals == al
        a_means[al] = float(np.mean(y[mask_a]))
        for bl in b_levels:
            mask = mask_a & (b_vals == bl)
            if np.any(mask):
                cell_means[(al, bl)] = float(np.mean(y[mask]))

    for bl in b_levels:
        mask_b = b_vals == bl
        b_means[bl] = float(np.mean(y[mask_b]))

    # SS decomposition
    ss_a = sum(np.sum(a_vals == al) * (a_means[al] - grand_mean) ** 2 for al in a_levels)
    ss_b = sum(np.sum(b_vals == bl) * (b_means[bl] - grand_mean) ** 2 for bl in b_levels)

    ss_total = float(np.sum((y - grand_mean) ** 2))

    # SS interaction
    ss_ab = 0.0
    for al in a_levels:
        for bl in b_levels:
            mask = (a_vals == al) & (b_vals == bl)
            n_cell = int(np.sum(mask))
            if n_cell > 0 and (al, bl) in cell_means:
                cell_effect = cell_means[(al, bl)] - a_means[al] - b_means[bl] + grand_mean
                ss_ab += n_cell * cell_effect ** 2

    ss_resid = ss_total - ss_a - ss_b - ss_ab

    df_a = len(a_levels) - 1
    df_b = len(b_levels) - 1
    df_ab = df_a * df_b
    df_resid = n - len(a_levels) * len(b_levels)

    if df_resid <= 0:
        df_resid = max(1, n - df_a - df_b - df_ab - 1)

    ms_a = ss_a / df_a if df_a > 0 else 0
    ms_b = ss_b / df_b if df_b > 0 else 0
    ms_ab = ss_ab / df_ab if df_ab > 0 else 0
    ms_resid = ss_resid / df_resid if df_resid > 0 else 0

    def _f_p(ms_effect, df_eff, ms_err, df_err):
        if ms_err <= 0:
            return 0.0, 1.0
        f = ms_effect / ms_err
        p = float(1 - stats.f.cdf(f, df_eff, df_err))
        return float(f), p

    f_a, p_a = _f_p(ms_a, df_a, ms_resid, df_resid)
    f_b, p_b = _f_p(ms_b, df_b, ms_resid, df_resid)
    f_ab, p_ab = _f_p(ms_ab, df_ab, ms_resid, df_resid)

    sources = [
        Anova2Source(source=factor_a, ss=float(ss_a), df=float(df_a), ms=float(ms_a),
                     f_statistic=f_a, p_value=p_a, partial_eta_sq=partial_eta_squared(ss_a, ss_resid)),
        Anova2Source(source=factor_b, ss=float(ss_b), df=float(df_b), ms=float(ms_b),
                     f_statistic=f_b, p_value=p_b, partial_eta_sq=partial_eta_squared(ss_b, ss_resid)),
        Anova2Source(source=f"{factor_a}:{factor_b}", ss=float(ss_ab), df=float(df_ab), ms=float(ms_ab),
                     f_statistic=f_ab, p_value=p_ab, partial_eta_sq=partial_eta_squared(ss_ab, ss_resid)),
    ]

    # Assumptions: normality of residuals + equal variance across cells
    residuals = []
    for al in a_levels:
        for bl in b_levels:
            mask = (a_vals == al) & (b_vals == bl)
            if np.any(mask) and (al, bl) in cell_means:
                residuals.extend((y[mask] - cell_means[(al, bl)]).tolist())

    assumptions = [check_normality(residuals, label="Residuals")]

    return Anova2Result(
        sources=sources,
        residual_df=float(df_resid),
        residual_ss=float(ss_resid),
        residual_ms=float(ms_resid),
        assumptions=assumptions,
    )
