"""Gage R&R — crossed and nested designs.

ANOVA-based variance decomposition for measurement system evaluation.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
from scipy import stats


@dataclass
class GageRRResult:
    """Gage R&R analysis result."""

    design: str  # "crossed" or "nested"
    var_repeatability: float = 0.0
    var_reproducibility: float = 0.0
    var_operator: float = 0.0
    var_interaction: float = 0.0
    var_part: float = 0.0
    var_gage_rr: float = 0.0
    var_total: float = 0.0
    pct_repeatability: float = 0.0
    pct_reproducibility: float = 0.0
    pct_gage_rr: float = 0.0
    pct_part: float = 0.0
    ndc: int = 0  # number of distinct categories
    n_operators: int = 0
    n_parts: int = 0
    n_replicates: int = 0
    anova_table: list[dict] = field(default_factory=list)


def crossed_gage_rr(
    measurements: list[float] | np.ndarray,
    parts: list[str | int],
    operators: list[str | int],
) -> GageRRResult:
    """Crossed Gage R&R (each operator measures each part).

    Uses two-way ANOVA with interaction to decompose variance.

    Args:
        measurements: All measurement values (flat array).
        parts: Part label for each measurement.
        operators: Operator label for each measurement.

    Returns:
        GageRRResult with variance components, %contribution, NDC.
    """
    y = np.asarray(measurements, dtype=float)
    p_arr = np.asarray(parts)
    o_arr = np.asarray(operators)
    n = len(y)

    unique_parts = sorted(set(parts))
    unique_ops = sorted(set(operators))
    n_parts = len(unique_parts)
    n_ops = len(unique_ops)

    # Determine replicates
    # Expect balanced design: n = n_parts * n_ops * n_reps
    n_reps = n // (n_parts * n_ops)
    if n_parts * n_ops * n_reps != n:
        raise ValueError(
            f"Unbalanced design: {n} measurements for {n_parts} parts × {n_ops} operators. "
            f"Expected {n_parts * n_ops} × r replicates."
        )

    grand_mean = float(np.mean(y))

    # Part means, operator means, cell means
    part_means = {}
    for p in unique_parts:
        part_means[p] = float(np.mean(y[p_arr == p]))

    op_means = {}
    for o in unique_ops:
        op_means[o] = float(np.mean(y[o_arr == o]))

    cell_means = {}
    for p in unique_parts:
        for o in unique_ops:
            mask = (p_arr == p) & (o_arr == o)
            if np.any(mask):
                cell_means[(p, o)] = float(np.mean(y[mask]))

    # Sum of squares
    ss_part = n_ops * n_reps * sum((m - grand_mean) ** 2 for m in part_means.values())
    ss_op = n_parts * n_reps * sum((m - grand_mean) ** 2 for m in op_means.values())

    ss_interaction = 0.0
    for p in unique_parts:
        for o in unique_ops:
            if (p, o) in cell_means:
                cell_eff = cell_means[(p, o)] - part_means[p] - op_means[o] + grand_mean
                ss_interaction += n_reps * cell_eff ** 2

    ss_total = float(np.sum((y - grand_mean) ** 2))
    ss_repeat = ss_total - ss_part - ss_op - ss_interaction

    # Degrees of freedom
    df_part = n_parts - 1
    df_op = n_ops - 1
    df_inter = df_part * df_op
    df_repeat = n_parts * n_ops * (n_reps - 1)

    # Mean squares
    ms_part = ss_part / df_part if df_part > 0 else 0
    ms_op = ss_op / df_op if df_op > 0 else 0
    ms_inter = ss_interaction / df_inter if df_inter > 0 else 0
    ms_repeat = ss_repeat / df_repeat if df_repeat > 0 else 0

    # Variance components (EMS method)
    var_repeat = ms_repeat
    var_inter = max(0, (ms_inter - ms_repeat) / n_reps)
    var_op = max(0, (ms_op - ms_inter) / (n_parts * n_reps))
    var_part = max(0, (ms_part - ms_inter) / (n_ops * n_reps))

    var_reprod = var_op + var_inter
    var_grr = var_repeat + var_reprod
    var_total = var_grr + var_part

    # Percentages
    if var_total > 0:
        pct_repeat = 100 * var_repeat / var_total
        pct_reprod = 100 * var_reprod / var_total
        pct_grr = 100 * var_grr / var_total
        pct_part = 100 * var_part / var_total
    else:
        pct_repeat = pct_reprod = pct_grr = pct_part = 0.0

    # NDC (number of distinct categories)
    ndc = int(1.41 * math.sqrt(var_part / var_grr)) if var_grr > 0 else 0

    # F-tests
    f_part = ms_part / ms_inter if ms_inter > 0 else 0
    f_op = ms_op / ms_inter if ms_inter > 0 else 0
    f_inter = ms_inter / ms_repeat if ms_repeat > 0 else 0

    anova_table = [
        {"source": "Part", "ss": float(ss_part), "df": df_part, "ms": float(ms_part),
         "f": float(f_part), "p": float(1 - stats.f.cdf(f_part, df_part, df_inter)) if df_inter > 0 else 1.0},
        {"source": "Operator", "ss": float(ss_op), "df": df_op, "ms": float(ms_op),
         "f": float(f_op), "p": float(1 - stats.f.cdf(f_op, df_op, df_inter)) if df_inter > 0 else 1.0},
        {"source": "Part×Operator", "ss": float(ss_interaction), "df": df_inter, "ms": float(ms_inter),
         "f": float(f_inter), "p": float(1 - stats.f.cdf(f_inter, df_inter, df_repeat)) if df_repeat > 0 else 1.0},
        {"source": "Repeatability", "ss": float(ss_repeat), "df": df_repeat, "ms": float(ms_repeat)},
        {"source": "Total", "ss": float(ss_total), "df": n - 1},
    ]

    return GageRRResult(
        design="crossed",
        var_repeatability=float(var_repeat),
        var_reproducibility=float(var_reprod),
        var_operator=float(var_op),
        var_interaction=float(var_inter),
        var_part=float(var_part),
        var_gage_rr=float(var_grr),
        var_total=float(var_total),
        pct_repeatability=float(pct_repeat),
        pct_reproducibility=float(pct_reprod),
        pct_gage_rr=float(pct_grr),
        pct_part=float(pct_part),
        ndc=ndc,
        n_operators=n_ops,
        n_parts=n_parts,
        n_replicates=n_reps,
        anova_table=anova_table,
    )
