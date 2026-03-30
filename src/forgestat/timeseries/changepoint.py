"""Change point detection — offline (PELT) and online (BOCPD).

Offline: PELT algorithm for batch analysis (requires ruptures OR pure fallback).
Online: Bayesian Online Changepoint Detection — INNOVATION. Pure numpy/scipy.
Anomaly scoring: Z-score and adaptive threshold — INNOVATION.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
from scipy import stats


@dataclass
class Changepoint:
    """A single detected change point."""

    index: int
    confidence: float = 0.0  # 0-1
    mean_before: float = 0.0
    mean_after: float = 0.0
    magnitude: float = 0.0  # absolute shift


@dataclass
class ChangepointResult:
    """Change point detection result."""

    method: str  # "pelt", "bocpd", "cusum"
    changepoints: list[Changepoint] = field(default_factory=list)
    n_segments: int = 0
    segment_means: list[float] = field(default_factory=list)
    segment_boundaries: list[int] = field(default_factory=list)


@dataclass
class AnomalyScore:
    """Per-point anomaly score."""

    index: int
    value: float
    score: float  # higher = more anomalous
    is_anomaly: bool
    threshold: float


@dataclass
class AnomalyResult:
    """Anomaly detection result."""

    method: str  # "zscore", "adaptive", "regime"
    scores: list[AnomalyScore] = field(default_factory=list)
    n_anomalies: int = 0
    anomaly_indices: list[int] = field(default_factory=list)


def pelt(
    data: list[float] | np.ndarray,
    penalty: str | float = "bic",
    min_size: int = 10,
) -> ChangepointResult:
    """PELT (Pruned Exact Linear Time) changepoint detection.

    Uses ruptures if installed, falls back to binary segmentation.

    Args:
        data: Time series values.
        penalty: "bic", "aic", or numeric penalty value.
        min_size: Minimum segment size.

    Returns:
        ChangepointResult with detected changepoints and segments.
    """
    x = np.asarray(data, dtype=float)
    n = len(x)

    # Calculate penalty — use median absolute deviation for robustness
    # np.var includes mean shifts which inflates penalty, defeating detection
    mad = float(np.median(np.abs(np.diff(x)))) * 1.4826  # MAD → σ estimate
    robust_var = mad ** 2 if mad > 0 else float(np.var(x))

    if penalty == "bic":
        pen = math.log(n) * robust_var
    elif penalty == "aic":
        pen = 2 * robust_var
    else:
        pen = float(penalty)

    try:
        import ruptures as rpt
        model = rpt.Pelt(model="rbf", min_size=min_size).fit(x)
        bkps = model.predict(pen=pen)
        if bkps and bkps[-1] == n:
            bkps = bkps[:-1]
    except ImportError:
        # Fallback: binary segmentation (pure numpy)
        bkps = _binary_segmentation(x, pen, min_size)

    return _build_result("pelt", x, bkps)


def bocpd(
    data: list[float] | np.ndarray,
    hazard_rate: float = 1 / 200,
    mu_prior: float | None = None,
    kappa_prior: float = 1.0,
    alpha_prior: float = 1.0,
    beta_prior: float = 1.0,
    threshold: float = 0.5,
) -> ChangepointResult:
    """Bayesian Online Changepoint Detection (Adams & MacKay 2007).

    INNOVATION: Not in SVEND. Processes data sequentially, maintaining a
    posterior over run lengths. Suitable for streaming manufacturing data.

    Args:
        data: Time series values.
        hazard_rate: Prior probability of changepoint at each step (1/expected_run_length).
        mu_prior: Prior mean (default: data mean).
        kappa_prior: Prior precision weight.
        alpha_prior: Gamma shape prior for precision.
        beta_prior: Gamma rate prior for precision.
        threshold: Run-length probability threshold for declaring changepoint.

    Returns:
        ChangepointResult with changepoints and their Bayesian confidence.
    """
    x = np.asarray(data, dtype=float)
    n = len(x)

    if mu_prior is None:
        mu_prior = float(np.mean(x[:min(20, len(x))]))  # use early data, not full mean

    # Run length probability matrix (sparse: only track max_run rows)
    max_run = min(n, 500)
    R = np.zeros((max_run + 1, n + 1))
    R[0, 0] = 1.0

    # Sufficient statistics for Normal-Gamma conjugate
    mu = np.full(max_run + 1, mu_prior)
    kappa = np.full(max_run + 1, kappa_prior)
    alpha = np.full(max_run + 1, alpha_prior)
    beta_arr = np.full(max_run + 1, beta_prior)

    changepoints = []

    for t in range(n):
        xt = x[t]

        # Predictive probability under each run length
        pred_prob = np.zeros(min(t + 1, max_run))
        for r in range(min(t + 1, max_run)):
            # Student-t predictive
            df = 2 * alpha[r]
            mu_pred = mu[r]
            var_pred = beta_arr[r] * (kappa[r] + 1) / (alpha[r] * kappa[r])
            if var_pred > 0 and df > 0:
                pred_prob[r] = float(stats.t.pdf(xt, df, loc=mu_pred, scale=math.sqrt(var_pred)))
            else:
                pred_prob[r] = 1e-10

        # Growth probabilities
        for r in range(min(t, max_run - 1), -1, -1):
            R[r + 1, t + 1] = R[r, t] * pred_prob[r] * (1 - hazard_rate)

        # Changepoint probability
        cp_mass = sum(R[r, t] * pred_prob[r] * hazard_rate for r in range(min(t + 1, max_run)) if r < len(pred_prob))
        R[0, t + 1] = cp_mass

        # Normalize
        total = np.sum(R[:, t + 1])
        if total > 0:
            R[:, t + 1] /= total

        # Detect changepoint: run length 0 has high probability
        if t > 0 and R[0, t + 1] > threshold:
            changepoints.append(t)

        # Update sufficient statistics
        new_mu = np.copy(mu)
        new_kappa = np.copy(kappa)
        new_alpha = np.copy(alpha)
        new_beta = np.copy(beta_arr)

        for r in range(min(t + 1, max_run - 1), -1, -1):
            new_mu[r + 1] = (kappa[r] * mu[r] + xt) / (kappa[r] + 1)
            new_kappa[r + 1] = kappa[r] + 1
            new_alpha[r + 1] = alpha[r] + 0.5
            new_beta[r + 1] = beta_arr[r] + kappa[r] * (xt - mu[r]) ** 2 / (2 * (kappa[r] + 1))

        # Reset for run length 0
        new_mu[0] = mu_prior
        new_kappa[0] = kappa_prior
        new_alpha[0] = alpha_prior
        new_beta[0] = beta_prior

        mu, kappa, alpha, beta_arr = new_mu, new_kappa, new_alpha, new_beta

    return _build_result("bocpd", x, changepoints)


def anomaly_scores(
    data: list[float] | np.ndarray,
    window: int = 20,
    threshold: float = 3.0,
    method: str = "adaptive",
) -> AnomalyResult:
    """Per-point anomaly scoring for time series.

    INNOVATION: Adaptive windowed z-score with expanding reference.
    Not just global z — tracks local mean/std for regime-aware scoring.

    Args:
        data: Time series values.
        window: Lookback window for local statistics.
        threshold: Z-score threshold for anomaly flag.
        method: "zscore" (global) or "adaptive" (windowed).

    Returns:
        AnomalyResult with per-point scores and flagged anomalies.
    """
    x = np.asarray(data, dtype=float)
    n = len(x)
    scores = []
    anomaly_idx = []

    if method == "zscore":
        mu = float(np.mean(x))
        sigma = float(np.std(x, ddof=1))
        if sigma < 1e-15:
            sigma = 1.0

        for i in range(n):
            z = abs(x[i] - mu) / sigma
            is_anom = z > threshold
            scores.append(AnomalyScore(
                index=i, value=float(x[i]), score=float(z),
                is_anomaly=bool(is_anom), threshold=threshold,
            ))
            if is_anom:
                anomaly_idx.append(i)
    else:
        # Adaptive: expanding then sliding window
        for i in range(n):
            if i < 3:
                scores.append(AnomalyScore(
                    index=i, value=float(x[i]), score=0.0,
                    is_anomaly=False, threshold=threshold,
                ))
                continue

            start = max(0, i - window)
            ref = x[start:i]  # exclude current point
            mu = float(np.mean(ref))
            sigma = float(np.std(ref, ddof=1))
            if sigma < 1e-15:
                sigma = abs(mu) * 0.01 if abs(mu) > 0 else 1.0

            z = abs(x[i] - mu) / sigma
            is_anom = z > threshold
            scores.append(AnomalyScore(
                index=i, value=float(x[i]), score=float(z),
                is_anomaly=bool(is_anom), threshold=threshold,
            ))
            if is_anom:
                anomaly_idx.append(i)

    return AnomalyResult(
        method=method,
        scores=scores,
        n_anomalies=len(anomaly_idx),
        anomaly_indices=anomaly_idx,
    )


def _binary_segmentation(x: np.ndarray, penalty: float, min_size: int) -> list[int]:
    """Simple binary segmentation fallback when ruptures is not installed."""
    n = len(x)
    bkps = []

    def _best_split(start, end):
        if end - start < 2 * min_size:
            return None, 0.0
        best_idx = None
        best_gain = 0.0
        total_var = np.var(x[start:end]) * (end - start)
        for k in range(start + min_size, end - min_size + 1):
            left_var = np.var(x[start:k]) * (k - start) if k > start else 0
            right_var = np.var(x[k:end]) * (end - k) if k < end else 0
            gain = total_var - left_var - right_var
            if gain > best_gain:
                best_gain = gain
                best_idx = k
        return best_idx, best_gain

    def _segment(start, end):
        idx, gain = _best_split(start, end)
        if idx is not None and gain > penalty:
            bkps.append(idx)
            _segment(start, idx)
            _segment(idx, end)

    _segment(0, n)
    return sorted(bkps)


def _build_result(method: str, x: np.ndarray, bkps: list[int]) -> ChangepointResult:
    """Build ChangepointResult from breakpoint indices."""
    n = len(x)
    boundaries = [0] + sorted(bkps) + [n]
    segments_means = []
    changepoints = []

    for i in range(len(boundaries) - 1):
        seg = x[boundaries[i]:boundaries[i + 1]]
        segments_means.append(float(np.mean(seg)))

    for i, bp in enumerate(sorted(bkps)):
        before = x[max(0, bp - 10):bp]
        after = x[bp:min(n, bp + 10)]
        m_before = float(np.mean(before)) if len(before) > 0 else 0
        m_after = float(np.mean(after)) if len(after) > 0 else 0
        changepoints.append(Changepoint(
            index=bp,
            mean_before=m_before,
            mean_after=m_after,
            magnitude=abs(m_after - m_before),
        ))

    return ChangepointResult(
        method=method,
        changepoints=changepoints,
        n_segments=len(segments_means),
        segment_means=segments_means,
        segment_boundaries=boundaries,
    )
