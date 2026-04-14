"""Concept drift and distribution shift detection.

Three detector families:
    ADWIN       — Adaptive windowing, detects changes in expectation of a bounded stream
    Page-Hinkley — Cumulative deviation from running mean, good for sustained shifts
    PSI          — Population Stability Index, discretized KL-like divergence

Dependencies: numpy, scipy.
"""

import numpy as np

__all__ = [
    "adwin_detect",
    "page_hinkley_detect",
    "compute_psi",
    "psi_severity",
    "PSI_LOW",
    "PSI_MODERATE",
    "PSI_HIGH",
]

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
PSI_BINS = 10
PSI_EPSILON = 1e-4
PSI_LOW = 0.10
PSI_MODERATE = 0.20
PSI_HIGH = 0.25
ADWIN_DELTA = 0.002
PH_DELTA = 0.005
PH_LAMBDA = 50


# ===========================================================================
# ADWIN — Adaptive Windowing
# ===========================================================================
def adwin_detect(stream, delta=ADWIN_DELTA, stream_name=""):
    """Simplified ADWIN: detect change in expectation of a bounded stream.

    Scans for the split point that maximizes evidence of mean change.
    Uses Hoeffding-style bound: eps = sqrt(1/(2m) * ln(4n/delta)) where m = min(n1,n2).

    Input stream is normalized to [0,1] for bounded guarantees.

    Parameters
    ----------
    stream : array-like
        Sequential observations.
    delta : float
        Confidence parameter (lower = fewer false alarms).
    stream_name : str
        Label for reporting.

    Returns
    -------
    dict with keys: detected, stream, change_idx, mean_before, mean_after,
                    shift_magnitude, window_length, epsilon (if detected).
    """
    x = np.asarray(stream, dtype=float)
    n = len(x)

    no_detect = {
        "detected": False,
        "stream": stream_name,
        "change_idx": None,
        "mean_before": None,
        "mean_after": None,
        "shift_magnitude": None,
        "window_length": n,
    }

    if n < 20:
        return no_detect

    x_min, x_max = float(np.min(x)), float(np.max(x))
    x_range = x_max - x_min
    if x_range < 1e-12:
        return no_detect

    x_norm = (x - x_min) / x_range
    cumsum = np.cumsum(x_norm)

    best_cut = None
    best_evidence = 0.0

    min_window = max(10, n // 20)
    for t in range(min_window, n - min_window):
        n1 = t
        n2 = n - t
        mean1 = cumsum[t - 1] / n1
        mean2 = (cumsum[-1] - cumsum[t - 1]) / n2

        m = min(n1, n2)
        epsilon = np.sqrt(0.5 / m * np.log(4 * n / delta))

        diff = abs(mean1 - mean2)
        evidence = diff - epsilon

        if evidence > best_evidence:
            best_evidence = evidence
            best_cut = t

    if best_cut is not None and best_evidence > 0:
        mean_before_raw = float(np.mean(x[:best_cut]))
        mean_after_raw = float(np.mean(x[best_cut:]))
        return {
            "detected": True,
            "stream": stream_name,
            "change_idx": int(best_cut),
            "mean_before": round(mean_before_raw, 6),
            "mean_after": round(mean_after_raw, 6),
            "shift_magnitude": round(mean_after_raw - mean_before_raw, 6),
            "window_length": n,
            "epsilon": round(
                float(np.sqrt(0.5 / min(best_cut, n - best_cut) * np.log(4 * n / delta))),
                6,
            ),
        }
    else:
        return {
            "detected": False,
            "stream": stream_name,
            "change_idx": None,
            "mean_before": round(float(np.mean(x)), 6),
            "mean_after": None,
            "shift_magnitude": None,
            "window_length": n,
        }


# ===========================================================================
# Page-Hinkley — Cumulative deviation detector
# ===========================================================================
def page_hinkley_detect(stream, direction="up", ph_delta=PH_DELTA, ph_lambda=PH_LAMBDA, stream_name=""):
    """Page-Hinkley test for sustained mean shift.

    For upward drift: track sum(x_t - x_bar_t - delta), alarm when max - current > lambda
    For downward drift: track sum(x_bar_t - x_t - delta), alarm when max - current > lambda

    Parameters
    ----------
    stream : array-like
        Sequential observations.
    direction : str
        "up" or "down".
    ph_delta : float
        Tolerance (minimum change to detect).
    ph_lambda : float
        Threshold for alarm.
    stream_name : str
        Label for reporting.

    Returns
    -------
    dict with keys: detected, direction, stream, change_idx, cumulative_value,
                    min_cumulative, delta, lambda.
    """
    x = np.asarray(stream, dtype=float)
    n = len(x)

    if n < 20:
        return {
            "detected": False,
            "direction": direction,
            "stream": stream_name,
            "change_idx": None,
            "cumulative_value": 0,
            "delta": ph_delta,
            "lambda": ph_lambda,
        }

    x_std = (x - np.mean(x)) / (np.std(x) + 1e-12)

    cumulative = 0.0
    min_cumulative = 0.0
    running_sum = 0.0
    change_idx = None

    for t in range(n):
        running_sum += x_std[t]
        running_mean = running_sum / (t + 1)

        if direction == "up":
            cumulative += x_std[t] - running_mean - ph_delta
        else:
            cumulative += running_mean - x_std[t] - ph_delta

        min_cumulative = min(min_cumulative, cumulative)

        if cumulative - min_cumulative > ph_lambda:
            change_idx = t
            break

    return {
        "detected": change_idx is not None,
        "direction": direction,
        "stream": stream_name,
        "change_idx": int(change_idx) if change_idx is not None else None,
        "cumulative_value": round(float(cumulative), 4),
        "min_cumulative": round(float(min_cumulative), 4),
        "delta": ph_delta,
        "lambda": ph_lambda,
    }


# ===========================================================================
# PSI — Population Stability Index
# ===========================================================================
def compute_psi(reference, current, n_bins=PSI_BINS):
    """Compute PSI between reference and current distributions.

    Binning: quantile-based from reference distribution.
    Smoothing: epsilon to avoid log(0).

    Parameters
    ----------
    reference : array-like
        Reference (baseline) distribution.
    current : array-like
        Current (comparison) distribution.
    n_bins : int
        Number of bins.

    Returns
    -------
    psi_value : float
        The PSI score.
    bin_contributions : list of dict
        Per-bin breakdown with ref_prop, cur_prop, psi_contribution.
    """
    ref = np.asarray(reference, dtype=float)
    cur = np.asarray(current, dtype=float)

    quantiles = np.linspace(0, 100, n_bins + 1)
    bin_edges = np.percentile(ref, quantiles)
    bin_edges = np.unique(bin_edges)
    if len(bin_edges) < 3:
        bin_edges = np.linspace(ref.min(), ref.max(), n_bins + 1)

    ref_counts, _ = np.histogram(ref, bins=bin_edges)
    cur_counts, _ = np.histogram(cur, bins=bin_edges)

    ref_props = ref_counts / len(ref) + PSI_EPSILON
    cur_props = cur_counts / len(cur) + PSI_EPSILON

    ref_props = ref_props / ref_props.sum()
    cur_props = cur_props / cur_props.sum()

    bin_psi = (cur_props - ref_props) * np.log(cur_props / ref_props)
    psi = float(np.sum(bin_psi))

    bin_contributions = [
        {
            "bin_low": float(bin_edges[i]),
            "bin_high": float(bin_edges[i + 1]),
            "ref_prop": float(ref_props[i]),
            "cur_prop": float(cur_props[i]),
            "psi_contribution": float(bin_psi[i]),
        }
        for i in range(len(bin_psi))
    ]

    return psi, bin_contributions


def psi_severity(psi_val):
    """Classify PSI value into severity category.

    Returns
    -------
    str : "high", "moderate", "low", or "negligible"
    """
    if psi_val >= PSI_HIGH:
        return "high"
    elif psi_val >= PSI_MODERATE:
        return "moderate"
    elif psi_val >= PSI_LOW:
        return "low"
    return "negligible"
