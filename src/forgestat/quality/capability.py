"""Process capability — attribute and non-normal methods.

Normal Cp/Cpk lives in forgespc. This module covers:
- Attribute capability (DPU, DPMO, sigma level)
- Non-normal capability (percentile-based Cnp/Cnpk)
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import stats


@dataclass
class AttributeCapability:
    """Attribute process capability result."""

    defects: int
    units: int
    opportunities: int
    dpu: float = 0.0
    dpo: float = 0.0
    dpmo: float = 0.0
    yield_pct: float = 0.0
    z_bench: float = 0.0
    sigma_short_term: float = 0.0  # with 1.5σ shift


@dataclass
class NonNormalCapability:
    """Non-normal (percentile-based) capability result."""

    cnp: float = 0.0  # equivalent of Cp
    cnpk: float = 0.0  # equivalent of Cpk
    median: float = 0.0
    p_low: float = 0.0  # 0.135th percentile
    p_high: float = 0.0  # 99.865th percentile
    ppm_out: float = 0.0
    is_normal: bool = False
    normality_p: float = 0.0


def attribute_capability(
    defects: int,
    units: int,
    opportunities: int = 1,
) -> AttributeCapability:
    """Compute attribute process capability.

    Args:
        defects: Total defects observed.
        units: Total units inspected.
        opportunities: Defect opportunities per unit.

    Returns:
        AttributeCapability with DPU, DPMO, sigma level.
    """
    total_opp = units * opportunities
    dpu = defects / units if units > 0 else 0
    dpo = defects / total_opp if total_opp > 0 else 0
    dpmo = dpo * 1_000_000

    yield_pct = (1 - dpo) * 100

    # Z-benchmark from defect rate
    z_bench = float(stats.norm.ppf(1 - dpo)) if 0 < dpo < 1 else (0.0 if dpo >= 1 else 6.0)
    sigma_st = z_bench + 1.5  # conventional 1.5σ shift

    return AttributeCapability(
        defects=defects,
        units=units,
        opportunities=opportunities,
        dpu=dpu,
        dpo=dpo,
        dpmo=dpmo,
        yield_pct=yield_pct,
        z_bench=z_bench,
        sigma_short_term=sigma_st,
    )


def nonnormal_capability(
    data: list[float] | np.ndarray,
    lsl: float | None = None,
    usl: float | None = None,
) -> NonNormalCapability:
    """Non-normal (percentile-based) process capability.

    Uses 0.135th and 99.865th percentiles (equivalent to ±3σ for normal)
    instead of assuming normality.

    Args:
        data: Process measurements.
        lsl: Lower specification limit (None if one-sided).
        usl: Upper specification limit (None if one-sided).

    Returns:
        NonNormalCapability with Cnp, Cnpk, PPM.
    """
    x = np.asarray(data, dtype=float)
    x = x[np.isfinite(x)]
    n = len(x)

    if n < 10:
        raise ValueError(f"Need at least 10 observations, got {n}")
    if lsl is None and usl is None:
        raise ValueError("At least one spec limit (lsl or usl) required")

    median = float(np.median(x))
    p_low = float(np.percentile(x, 0.135))
    p_high = float(np.percentile(x, 99.865))

    # Normality test
    ad_result = stats.anderson(x, dist="norm")
    cv = ad_result.critical_values[2] if len(ad_result.critical_values) > 2 else 1.0
    is_normal = bool(ad_result.statistic < cv)

    # Non-normal indices
    spread = p_high - p_low
    cnp = 0.0
    cnpk = 0.0

    if lsl is not None and usl is not None and spread > 0:
        cnp = (usl - lsl) / spread
        cnpk_upper = (usl - median) / (p_high - median) if p_high > median else 0
        cnpk_lower = (median - lsl) / (median - p_low) if median > p_low else 0
        cnpk = min(cnpk_upper, cnpk_lower)
    elif usl is not None and p_high > median:
        cnpk = (usl - median) / (p_high - median)
        cnp = cnpk
    elif lsl is not None and median > p_low:
        cnpk = (median - lsl) / (median - p_low)
        cnp = cnpk

    # PPM outside spec
    n_out = 0
    if lsl is not None:
        n_out += int(np.sum(x < lsl))
    if usl is not None:
        n_out += int(np.sum(x > usl))
    ppm_out = n_out / n * 1_000_000

    return NonNormalCapability(
        cnp=cnp,
        cnpk=cnpk,
        median=median,
        p_low=p_low,
        p_high=p_high,
        ppm_out=ppm_out,
        is_normal=is_normal,
        normality_p=float(ad_result.statistic),
    )
