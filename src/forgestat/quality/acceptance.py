"""Acceptance sampling — attribute and variable plans.

AQL/LTPD-based sample size, OC curves, AOQ/ATI.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
from scipy import stats


@dataclass
class SamplingPlan:
    """Acceptance sampling plan result."""

    plan_type: str  # "attribute" or "variable"
    sample_size: int = 0
    acceptance_number: int = 0  # for attribute plans
    k_value: float = 0.0  # for variable plans
    aql: float = 0.0
    ltpd: float = 0.0
    oc_curve: list[tuple[float, float]] = field(default_factory=list)  # [(defect_rate, P(accept))]
    aoql: float | None = None
    producer_risk: float = 0.0  # P(reject | quality = AQL)
    consumer_risk: float = 0.0  # P(accept | quality = LTPD)


def attribute_plan(
    aql: float = 0.01,
    ltpd: float = 0.05,
    producer_risk: float = 0.05,
    consumer_risk: float = 0.10,
    lot_size: int = 1000,
) -> SamplingPlan:
    """Design a single-sampling attribute plan.

    Finds (n, c) such that:
    - P(accept | p=AQL) ≥ 1 - producer_risk
    - P(accept | p=LTPD) ≤ consumer_risk

    Args:
        aql: Acceptable Quality Level (fraction defective).
        ltpd: Lot Tolerance Percent Defective.
        producer_risk: α (Type I error at AQL).
        consumer_risk: β (Type II error at LTPD).
        lot_size: Lot size (for AOQ/ATI calculations).
    """
    # Search for minimum (n, c)
    best_n = lot_size
    best_c = 0

    for c in range(0, 50):
        for n in range(c + 1, min(lot_size + 1, 5000)):
            pa_aql = float(stats.binom.cdf(c, n, aql))
            pa_ltpd = float(stats.binom.cdf(c, n, ltpd))

            if pa_aql >= 1 - producer_risk and pa_ltpd <= consumer_risk:
                if n < best_n:
                    best_n = n
                    best_c = c
                break
        if best_n < lot_size:
            break

    # OC curve
    oc = []
    aoq_vals = []
    defect_rates = np.linspace(0, min(0.2, ltpd * 3), 50)
    for p in defect_rates:
        pa = float(stats.binom.cdf(best_c, best_n, max(p, 1e-10)))
        oc.append((float(p), pa))
        if lot_size > best_n:
            aoq = pa * p * (lot_size - best_n) / lot_size
            aoq_vals.append(aoq)

    aoql = max(aoq_vals) if aoq_vals else None

    pa_aql = float(stats.binom.cdf(best_c, best_n, aql))
    pa_ltpd = float(stats.binom.cdf(best_c, best_n, ltpd))

    return SamplingPlan(
        plan_type="attribute",
        sample_size=best_n,
        acceptance_number=best_c,
        aql=aql,
        ltpd=ltpd,
        oc_curve=oc,
        aoql=aoql,
        producer_risk=1 - pa_aql,
        consumer_risk=pa_ltpd,
    )


def variable_plan(
    aql: float = 0.01,
    ltpd: float = 0.05,
    producer_risk: float = 0.05,
    consumer_risk: float = 0.10,
) -> SamplingPlan:
    """Design a variables acceptance sampling plan (k-method).

    Assumes known sigma and normal distribution.

    Args:
        aql: Acceptable Quality Level.
        ltpd: Lot Tolerance Percent Defective.
        producer_risk: α.
        consumer_risk: β.
    """
    z_aql = stats.norm.ppf(1 - aql)
    z_ltpd = stats.norm.ppf(1 - ltpd)
    z_alpha = stats.norm.ppf(1 - producer_risk)
    z_beta = stats.norm.ppf(1 - consumer_risk)

    # Sample size
    n = math.ceil(((z_alpha + z_beta) / (z_aql - z_ltpd)) ** 2 + 0.5 * z_alpha ** 2)
    n = max(2, n)

    # k-value (acceptance constant)
    k = z_aql - z_alpha / math.sqrt(n)

    # OC curve
    oc = []
    defect_rates = np.linspace(0.001, min(0.2, ltpd * 3), 50)
    for p in defect_rates:
        z_p = stats.norm.ppf(1 - p)
        pa = float(stats.norm.cdf((z_p - k) * math.sqrt(n)))
        oc.append((float(p), pa))

    return SamplingPlan(
        plan_type="variable",
        sample_size=n,
        k_value=k,
        aql=aql,
        ltpd=ltpd,
        oc_curve=oc,
        producer_risk=producer_risk,
        consumer_risk=consumer_risk,
    )
