"""Meta-analysis — fixed and random effects pooling.

Inverse-variance weighting, DerSimonian-Laird estimator, heterogeneity.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
from scipy import stats


@dataclass
class StudyEffect:
    """One study in a meta-analysis."""

    name: str
    effect: float
    se: float
    weight: float = 0.0
    ci_lower: float = 0.0
    ci_upper: float = 0.0


@dataclass
class MetaAnalysisResult:
    """Meta-analysis result."""

    model: str  # "fixed" or "random"
    pooled_effect: float = 0.0
    pooled_se: float = 0.0
    pooled_ci_lower: float = 0.0
    pooled_ci_upper: float = 0.0
    pooled_z: float = 0.0
    pooled_p: float = 0.0
    studies: list[StudyEffect] = field(default_factory=list)
    q_statistic: float = 0.0  # Cochran's Q
    q_p_value: float = 0.0
    i_squared: float = 0.0  # I² heterogeneity
    tau_squared: float = 0.0  # between-study variance (random effects)
    k: int = 0  # number of studies


def meta_analysis(
    effects: list[float],
    standard_errors: list[float],
    study_names: list[str] | None = None,
    model: str = "random",
    ci_level: float = 0.95,
) -> MetaAnalysisResult:
    """Fixed or random effects meta-analysis.

    Args:
        effects: Point estimates from each study (e.g., mean differences, log-ORs).
        standard_errors: Standard errors of the estimates.
        study_names: Names for each study.
        model: "fixed" or "random" (DerSimonian-Laird).
        ci_level: Confidence level.

    Returns:
        MetaAnalysisResult with pooled effect, heterogeneity statistics.
    """
    y = np.asarray(effects, dtype=float)
    se = np.asarray(standard_errors, dtype=float)
    k = len(y)
    names = study_names or [f"Study {i+1}" for i in range(k)]
    z_crit = stats.norm.ppf((1 + ci_level) / 2)

    if k < 2:
        raise ValueError("Need at least 2 studies for meta-analysis")

    # Fixed-effect weights (inverse variance)
    w_fe = 1 / (se ** 2)
    theta_fe = float(np.sum(w_fe * y) / np.sum(w_fe))

    # Cochran's Q
    Q = float(np.sum(w_fe * (y - theta_fe) ** 2))
    q_df = k - 1
    q_p = float(1 - stats.chi2.cdf(Q, q_df)) if q_df > 0 else 1.0

    # I² heterogeneity
    i_sq = max(0, (Q - q_df) / Q * 100) if Q > 0 else 0.0

    # DerSimonian-Laird tau²
    C = float(np.sum(w_fe) - np.sum(w_fe ** 2) / np.sum(w_fe))
    tau2 = max(0, (Q - q_df) / C) if C > 0 else 0.0

    if model == "random":
        w = 1 / (se ** 2 + tau2)
    else:
        w = w_fe
        tau2 = 0.0

    theta = float(np.sum(w * y) / np.sum(w))
    se_theta = float(1 / math.sqrt(np.sum(w)))
    z_stat = theta / se_theta if se_theta > 0 else 0.0
    p_val = float(2 * (1 - stats.norm.cdf(abs(z_stat))))

    ci_lo = theta - z_crit * se_theta
    ci_hi = theta + z_crit * se_theta

    # Per-study details
    studies = []
    for i in range(k):
        s_ci_lo = float(y[i] - z_crit * se[i])
        s_ci_hi = float(y[i] + z_crit * se[i])
        studies.append(StudyEffect(
            name=names[i],
            effect=float(y[i]),
            se=float(se[i]),
            weight=float(w[i] / np.sum(w) * 100),
            ci_lower=s_ci_lo,
            ci_upper=s_ci_hi,
        ))

    return MetaAnalysisResult(
        model=model,
        pooled_effect=theta,
        pooled_se=se_theta,
        pooled_ci_lower=float(ci_lo),
        pooled_ci_upper=float(ci_hi),
        pooled_z=z_stat,
        pooled_p=p_val,
        studies=studies,
        q_statistic=Q,
        q_p_value=q_p,
        i_squared=i_sq,
        tau_squared=tau2,
        k=k,
    )
