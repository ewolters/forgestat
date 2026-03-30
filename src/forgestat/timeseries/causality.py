"""Granger causality and regime detection.

Granger: does X's past help predict Y?
Regime detection: hidden state switching — INNOVATION.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class GrangerResult:
    """Granger causality test result."""

    x_causes_y: bool = False
    results_by_lag: list[dict] = field(default_factory=list)  # [{lag, f_stat, p_value}]
    best_lag: int = 0
    best_p_value: float = 1.0
    max_lag_tested: int = 0


@dataclass
class RegimeState:
    """One detected regime in a time series."""

    start: int
    end: int
    mean: float
    std: float
    label: str = ""  # "stable", "volatile", "trending_up", "trending_down"


@dataclass
class RegimeResult:
    """Regime detection result."""

    n_regimes: int = 0
    regimes: list[RegimeState] = field(default_factory=list)
    regime_labels: list[str] = field(default_factory=list)  # per-point label


def granger_causality(
    x: list[float] | np.ndarray,
    y: list[float] | np.ndarray,
    max_lag: int = 4,
    alpha: float = 0.05,
) -> GrangerResult:
    """Granger causality test: does X help predict Y?

    H₀: Past values of X do not improve prediction of Y.

    Args:
        x: Potential cause series.
        y: Effect series.
        max_lag: Maximum number of lags to test.
        alpha: Significance level.

    Returns:
        GrangerResult with F-stats and p-values per lag.
    """
    from statsmodels.tsa.stattools import grangercausalitytests

    x_arr = np.asarray(x, dtype=float)
    y_arr = np.asarray(y, dtype=float)
    n = min(len(x_arr), len(y_arr))
    x_arr = x_arr[:n]
    y_arr = y_arr[:n]

    # grangercausalitytests expects: column 0 = y (effect), column 1 = x (cause)
    data_matrix = np.column_stack([y_arr, x_arr])

    max_lag = min(max_lag, n // 3)
    if max_lag < 1:
        return GrangerResult()

    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        gc = grangercausalitytests(data_matrix, maxlag=max_lag, verbose=False)

    results = []
    best_lag = 0
    best_p = 1.0

    for lag in range(1, max_lag + 1):
        f_stat, p_val = gc[lag][0]["ssr_ftest"][:2]
        results.append({"lag": lag, "f_stat": float(f_stat), "p_value": float(p_val)})
        if p_val < best_p:
            best_p = float(p_val)
            best_lag = lag

    return GrangerResult(
        x_causes_y=best_p < alpha,
        results_by_lag=results,
        best_lag=best_lag,
        best_p_value=best_p,
        max_lag_tested=max_lag,
    )


def detect_regimes(
    data: list[float] | np.ndarray,
    n_regimes: int = 2,
    min_segment: int = 10,
) -> RegimeResult:
    """Regime detection via Gaussian mixture on windowed statistics.

    INNOVATION: Not in SVEND. Classifies each segment by its statistical
    character (stable/volatile/trending) rather than just mean shifts.
    Useful for identifying process states in manufacturing.

    Args:
        data: Time series values.
        n_regimes: Expected number of regimes (2-4).
        min_segment: Minimum segment length for feature computation.

    Returns:
        RegimeResult with regime boundaries and labels.
    """
    x = np.asarray(data, dtype=float)
    n = len(x)

    if n < min_segment * 2:
        return RegimeResult(
            n_regimes=1,
            regimes=[RegimeState(start=0, end=n, mean=float(np.mean(x)), std=float(np.std(x, ddof=1)))],
            regime_labels=["stable"] * n,
        )

    # Compute windowed features: mean, std, trend slope
    window = max(min_segment, n // 20)
    features = []
    for i in range(0, n - window + 1, max(1, window // 2)):
        seg = x[i:i + window]
        m = float(np.mean(seg))
        s = float(np.std(seg, ddof=1))
        # Simple trend: slope of linear fit
        t = np.arange(len(seg))
        slope = float(np.polyfit(t, seg, 1)[0]) if len(seg) > 1 else 0.0
        features.append((i, i + window, m, s, slope))

    if not features:
        return RegimeResult(n_regimes=1, regimes=[], regime_labels=[])

    # Cluster features by (normalized mean, normalized std, normalized slope)
    feat_arr = np.array([[f[2], f[3], f[4]] for f in features])
    # Normalize each dimension
    for col in range(feat_arr.shape[1]):
        rng = feat_arr[:, col].max() - feat_arr[:, col].min()
        if rng > 1e-15:
            feat_arr[:, col] = (feat_arr[:, col] - feat_arr[:, col].min()) / rng

    # K-means clustering (pure numpy, no sklearn)
    labels = _kmeans(feat_arr, n_regimes)

    # Map labels to regime types
    regime_types = {}
    for k in range(n_regimes):
        mask = labels == k
        if not np.any(mask):
            continue
        cluster_feats = np.array([[features[i][2], features[i][3], features[i][4]] for i in range(len(features)) if mask[i]])
        avg_std = float(np.mean(cluster_feats[:, 1]))
        avg_slope = float(np.mean(cluster_feats[:, 2]))
        overall_std = float(np.std(x, ddof=1))

        if abs(avg_slope) > overall_std * 0.1:
            regime_types[k] = "trending_up" if avg_slope > 0 else "trending_down"
        elif avg_std > overall_std * 0.5:
            regime_types[k] = "volatile"
        else:
            regime_types[k] = "stable"

    # Build per-point labels
    point_labels = ["stable"] * n
    for i, lab in enumerate(labels):
        start = features[i][0]
        end = features[i][1]
        for j in range(start, min(end, n)):
            point_labels[j] = regime_types.get(lab, "stable")

    # Build regime segments (consecutive same-label runs)
    regimes = []
    current_label = point_labels[0]
    seg_start = 0
    for i in range(1, n):
        if point_labels[i] != current_label:
            seg = x[seg_start:i]
            regimes.append(RegimeState(
                start=seg_start, end=i,
                mean=float(np.mean(seg)), std=float(np.std(seg, ddof=1)) if len(seg) > 1 else 0,
                label=current_label,
            ))
            current_label = point_labels[i]
            seg_start = i

    seg = x[seg_start:n]
    regimes.append(RegimeState(
        start=seg_start, end=n,
        mean=float(np.mean(seg)), std=float(np.std(seg, ddof=1)) if len(seg) > 1 else 0,
        label=current_label,
    ))

    return RegimeResult(
        n_regimes=len(set(point_labels)),
        regimes=regimes,
        regime_labels=point_labels,
    )


def _kmeans(X: np.ndarray, k: int, max_iter: int = 50) -> np.ndarray:
    """Simple k-means (pure numpy, no sklearn)."""
    n = len(X)
    if n <= k:
        return np.arange(n)

    rng = np.random.default_rng(42)
    centers = X[rng.choice(n, k, replace=False)]

    labels = np.zeros(n, dtype=int)
    for _ in range(max_iter):
        # Assign
        dists = np.array([[np.sum((x - c) ** 2) for c in centers] for x in X])
        new_labels = np.argmin(dists, axis=1)
        if np.array_equal(new_labels, labels):
            break
        labels = new_labels
        # Update centers
        for j in range(k):
            mask = labels == j
            if np.any(mask):
                centers[j] = np.mean(X[mask], axis=0)

    return labels
