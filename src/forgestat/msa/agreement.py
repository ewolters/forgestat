"""Agreement analysis — ICC, Bland-Altman, linearity/bias.

Measurement method comparison and calibration assessment.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
from scipy import stats


@dataclass
class ICCResult:
    """Intraclass Correlation Coefficient result."""

    icc: float = 0.0
    icc_type: str = ""  # "ICC(1,1)", "ICC(2,1)", "ICC(3,1)"
    ci_lower: float = 0.0
    ci_upper: float = 0.0
    f_statistic: float = 0.0
    p_value: float = 0.0
    n_subjects: int = 0
    n_raters: int = 0


@dataclass
class BlandAltmanResult:
    """Bland-Altman method comparison result."""

    mean_diff: float = 0.0  # bias
    std_diff: float = 0.0
    loa_lower: float = 0.0  # lower limit of agreement
    loa_upper: float = 0.0  # upper limit of agreement
    ci_mean_lower: float = 0.0
    ci_mean_upper: float = 0.0
    n: int = 0
    proportional_bias: bool = False  # slope of diff vs mean significant?


@dataclass
class LinearityBiasResult:
    """Gage linearity and bias result."""

    overall_bias: float = 0.0
    linearity_slope: float = 0.0
    linearity_intercept: float = 0.0
    linearity_r_squared: float = 0.0
    linearity_p_value: float = 0.0
    bias_per_level: dict[float, float] = field(default_factory=dict)
    bias_significant: bool = False


def icc(
    ratings: list[list[float]] | np.ndarray,
    icc_type: str = "ICC(3,1)",
    ci_level: float = 0.95,
) -> ICCResult:
    """Intraclass Correlation Coefficient.

    Args:
        ratings: n_subjects × n_raters matrix. Each row is one subject,
                 each column is one rater's measurement.
        icc_type: "ICC(1,1)", "ICC(2,1)", or "ICC(3,1)".
            - ICC(1,1): One-way random, single measures
            - ICC(2,1): Two-way random, single measures (absolute agreement)
            - ICC(3,1): Two-way mixed, single measures (consistency)
        ci_level: Confidence level.

    Returns:
        ICCResult with ICC value, CI, F-test.
    """
    Y = np.asarray(ratings, dtype=float)
    n, k = Y.shape  # n subjects, k raters

    # Mean squares from two-way ANOVA
    grand_mean = np.mean(Y)
    row_means = np.mean(Y, axis=1)
    col_means = np.mean(Y, axis=0)

    ss_rows = k * np.sum((row_means - grand_mean) ** 2)
    ss_cols = n * np.sum((col_means - grand_mean) ** 2)
    ss_total = np.sum((Y - grand_mean) ** 2)
    ss_error = ss_total - ss_rows - ss_cols

    ms_rows = ss_rows / (n - 1) if n > 1 else 0
    ms_cols = ss_cols / (k - 1) if k > 1 else 0
    ms_error = ss_error / ((n - 1) * (k - 1)) if n > 1 and k > 1 else 0

    # ICC computation
    if icc_type == "ICC(1,1)":
        # One-way random
        ms_within = (ss_total - ss_rows) / (n * (k - 1)) if n > 0 and k > 1 else 0
        icc_val = (ms_rows - ms_within) / (ms_rows + (k - 1) * ms_within) if ms_rows + (k - 1) * ms_within > 0 else 0
    elif icc_type == "ICC(2,1)":
        # Two-way random, absolute agreement
        denom = ms_rows + (k - 1) * ms_error + k * (ms_cols - ms_error) / n
        icc_val = (ms_rows - ms_error) / denom if denom > 0 else 0
    else:  # ICC(3,1)
        # Two-way mixed, consistency
        denom = ms_rows + (k - 1) * ms_error
        icc_val = (ms_rows - ms_error) / denom if denom > 0 else 0

    icc_val = float(max(0, min(1, icc_val)))

    # F-test for ICC
    f_stat = ms_rows / ms_error if ms_error > 0 else 0
    df1 = n - 1
    df2 = (n - 1) * (k - 1)
    p_value = float(1 - stats.f.cdf(f_stat, df1, df2)) if df2 > 0 else 1.0

    # CI (Shrout & Fleiss approximation for ICC(3,1))
    alpha = 1 - ci_level
    f_lo = stats.f.ppf(alpha / 2, df1, df2) if df2 > 0 else 1
    f_hi = stats.f.ppf(1 - alpha / 2, df1, df2) if df2 > 0 else 1

    if f_hi > 0:
        ci_lower = max(0, (f_stat / f_hi - 1) / (f_stat / f_hi + k - 1))
    else:
        ci_lower = 0.0
    if f_lo > 0:
        ci_upper = min(1, (f_stat / f_lo - 1) / (f_stat / f_lo + k - 1))
    else:
        ci_upper = 1.0

    return ICCResult(
        icc=icc_val,
        icc_type=icc_type,
        ci_lower=float(ci_lower),
        ci_upper=float(ci_upper),
        f_statistic=float(f_stat),
        p_value=p_value,
        n_subjects=n,
        n_raters=k,
    )


def bland_altman(
    method1: list[float] | np.ndarray,
    method2: list[float] | np.ndarray,
    ci_level: float = 0.95,
) -> BlandAltmanResult:
    """Bland-Altman method comparison (limits of agreement).

    Plots difference vs mean of two measurement methods.

    Args:
        method1: Measurements from method 1.
        method2: Measurements from method 2 (paired).
        ci_level: Confidence level for mean bias CI.

    Returns:
        BlandAltmanResult with bias, limits of agreement, proportional bias flag.
    """
    m1 = np.asarray(method1, dtype=float)
    m2 = np.asarray(method2, dtype=float)

    if len(m1) != len(m2):
        raise ValueError("Methods must have same number of measurements")

    diff = m1 - m2
    mean_pair = (m1 + m2) / 2
    n = len(diff)

    mean_diff = float(np.mean(diff))
    std_diff = float(np.std(diff, ddof=1))

    loa_lower = mean_diff - 1.96 * std_diff
    loa_upper = mean_diff + 1.96 * std_diff

    se_mean = std_diff / math.sqrt(n)
    t_crit = stats.t.ppf((1 + ci_level) / 2, n - 1)
    ci_lo = mean_diff - t_crit * se_mean
    ci_hi = mean_diff + t_crit * se_mean

    # Check proportional bias (regression of diff on mean)
    slope, intercept, r, p, se = stats.linregress(mean_pair, diff)
    proportional = p < 0.05

    return BlandAltmanResult(
        mean_diff=mean_diff,
        std_diff=std_diff,
        loa_lower=float(loa_lower),
        loa_upper=float(loa_upper),
        ci_mean_lower=float(ci_lo),
        ci_mean_upper=float(ci_hi),
        n=n,
        proportional_bias=proportional,
    )


def linearity_bias(
    reference: list[float] | np.ndarray,
    measured: list[float] | np.ndarray,
) -> LinearityBiasResult:
    """Gage linearity and bias assessment.

    Regresses bias (measured - reference) on reference value.
    Linearity = slope; Bias = intercept + overall mean bias.

    Args:
        reference: Known/reference values.
        measured: Gage/instrument readings.

    Returns:
        LinearityBiasResult with linearity slope, overall bias, per-level biases.
    """
    ref = np.asarray(reference, dtype=float)
    meas = np.asarray(measured, dtype=float)

    if len(ref) != len(meas):
        raise ValueError("Reference and measured must have same length")

    bias = meas - ref
    overall_bias = float(np.mean(bias))

    slope, intercept, r, p, se = stats.linregress(ref, bias)

    # Per-level bias (unique reference values)
    unique_refs = sorted(set(ref.tolist()))
    bias_per_level = {}
    for rv in unique_refs:
        mask = ref == rv
        bias_per_level[rv] = float(np.mean(bias[mask]))

    return LinearityBiasResult(
        overall_bias=overall_bias,
        linearity_slope=float(slope),
        linearity_intercept=float(intercept),
        linearity_r_squared=float(r ** 2),
        linearity_p_value=float(p),
        bias_per_level=bias_per_level,
        bias_significant=p < 0.05,
    )
