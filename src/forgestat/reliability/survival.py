"""Survival analysis — Kaplan-Meier estimator and log-rank test.

Pure numpy/scipy. No lifelines or statsmodels required.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
from scipy import stats


@dataclass
class SurvivalPoint:
    """One step in a Kaplan-Meier curve."""

    time: float
    at_risk: int
    events: int
    censored: int
    survival: float
    se: float = 0.0
    ci_lower: float = 0.0
    ci_upper: float = 0.0


@dataclass
class KaplanMeierResult:
    """Kaplan-Meier survival analysis result."""

    curve: list[SurvivalPoint] = field(default_factory=list)
    median_survival: float | None = None
    mean_survival: float = 0.0
    n_events: int = 0
    n_censored: int = 0
    n_total: int = 0


@dataclass
class LogRankResult:
    """Log-rank test comparing two survival curves."""

    chi_square: float = 0.0
    p_value: float = 0.0
    df: int = 1
    observed: dict[str, int] = field(default_factory=dict)  # {group: observed events}
    expected: dict[str, float] = field(default_factory=dict)  # {group: expected events}


def kaplan_meier(
    times: list[float] | np.ndarray,
    events: list[bool] | np.ndarray | None = None,
    ci_level: float = 0.95,
) -> KaplanMeierResult:
    """Kaplan-Meier product-limit survival estimator.

    Args:
        times: Observed times (failure or censoring).
        events: True = failure, False = censored. If None, all are failures.
        ci_level: Confidence level for Greenwood CI.

    Returns:
        KaplanMeierResult with survival curve, median, mean.
    """
    t = np.asarray(times, dtype=float)
    n = len(t)

    if events is None:
        e = np.ones(n, dtype=bool)
    else:
        e = np.asarray(events, dtype=bool)

    if len(e) != n:
        raise ValueError("times and events must have same length")

    n_events = int(np.sum(e))
    n_censored = n - n_events

    # Sort by time (censored after failures at same time)
    order = np.lexsort((~e, t))
    t_sorted = t[order]
    e_sorted = e[order]

    # Unique event times
    unique_times = np.unique(t_sorted[e_sorted])

    curve = []
    survival = 1.0
    var_sum = 0.0
    at_risk = n
    z = stats.norm.ppf((1 + ci_level) / 2)

    # Add time=0 point
    curve.append(SurvivalPoint(time=0.0, at_risk=n, events=0, censored=0,
                                survival=1.0, se=0.0, ci_lower=1.0, ci_upper=1.0))

    for time_i in unique_times:
        # Count censored before this time
        cens_before = int(np.sum((t_sorted < time_i) & ~e_sorted & (t_sorted > (curve[-1].time if curve else 0))))
        at_risk -= cens_before

        # Events at this time
        d_i = int(np.sum((t_sorted == time_i) & e_sorted))
        c_i = int(np.sum((t_sorted == time_i) & ~e_sorted))

        if at_risk > 0 and d_i > 0:
            survival *= (1 - d_i / at_risk)
            # Greenwood variance
            if at_risk > d_i:
                var_sum += d_i / (at_risk * (at_risk - d_i))

        se = survival * math.sqrt(var_sum) if var_sum >= 0 else 0
        ci_lo = max(0, survival - z * se)
        ci_hi = min(1, survival + z * se)

        curve.append(SurvivalPoint(
            time=float(time_i), at_risk=at_risk, events=d_i, censored=c_i,
            survival=float(survival), se=float(se),
            ci_lower=float(ci_lo), ci_upper=float(ci_hi),
        ))

        at_risk -= (d_i + c_i)

    # Median survival: first time S(t) <= 0.5
    median = None
    for pt in curve:
        if pt.survival <= 0.5:
            median = pt.time
            break

    # Restricted mean (area under curve up to last observed time)
    mean_surv = 0.0
    for i in range(1, len(curve)):
        dt = curve[i].time - curve[i - 1].time
        mean_surv += curve[i - 1].survival * dt

    return KaplanMeierResult(
        curve=curve,
        median_survival=median,
        mean_survival=float(mean_surv),
        n_events=n_events,
        n_censored=n_censored,
        n_total=n,
    )


def log_rank_test(
    times1: list[float] | np.ndarray,
    events1: list[bool] | np.ndarray,
    times2: list[float] | np.ndarray,
    events2: list[bool] | np.ndarray,
    group_names: tuple[str, str] = ("Group 1", "Group 2"),
) -> LogRankResult:
    """Log-rank (Mantel-Haenszel) test comparing two survival curves.

    H₀: The two groups have the same survival function.

    Args:
        times1, events1: Times and event indicators for group 1.
        times2, events2: Times and event indicators for group 2.
        group_names: Names for the two groups.

    Returns:
        LogRankResult with chi-square statistic and p-value.
    """
    t1 = np.asarray(times1, dtype=float)
    e1 = np.asarray(events1, dtype=bool)
    t2 = np.asarray(times2, dtype=float)
    e2 = np.asarray(events2, dtype=bool)

    # Combine and get unique event times
    all_times = np.concatenate([t1[e1], t2[e2]])
    unique_times = np.sort(np.unique(all_times))

    O1 = 0  # observed events group 1
    E1 = 0.0  # expected events group 1
    V = 0.0  # variance

    n1_at_risk = len(t1)
    n2_at_risk = len(t2)

    for ti in unique_times:
        d1 = int(np.sum((t1 == ti) & e1))
        d2 = int(np.sum((t2 == ti) & e2))
        d_total = d1 + d2
        n_total = n1_at_risk + n2_at_risk

        if n_total > 0:
            e1_i = n1_at_risk * d_total / n_total
            E1 += e1_i
            O1 += d1

            # Hypergeometric variance
            if n_total > 1:
                V += (n1_at_risk * n2_at_risk * d_total * (n_total - d_total)) / (n_total ** 2 * (n_total - 1))

        # Remove events and censored at this time
        n1_at_risk -= int(np.sum(t1 == ti))
        n2_at_risk -= int(np.sum(t2 == ti))

    chi2 = (O1 - E1) ** 2 / V if V > 0 else 0.0
    p_val = float(1 - stats.chi2.cdf(chi2, 1))

    O2 = int(np.sum(e2))
    E2_total = float(np.sum(e1) + np.sum(e2)) - E1

    return LogRankResult(
        chi_square=float(chi2),
        p_value=p_val,
        df=1,
        observed={group_names[0]: O1, group_names[1]: O2},
        expected={group_names[0]: float(E1), group_names[1]: E2_total},
    )
