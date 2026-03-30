"""Autocorrelation, partial autocorrelation, cross-correlation, Ljung-Box.

ACF/PACF for ARIMA order identification. CCF for lead-lag relationships.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class ACFResult:
    """ACF/PACF analysis result."""

    acf_values: list[float] = field(default_factory=list)
    pacf_values: list[float] = field(default_factory=list)
    n_lags: int = 0
    confidence_bound: float = 0.0  # ±1.96/√n
    significant_acf_lags: list[int] = field(default_factory=list)
    significant_pacf_lags: list[int] = field(default_factory=list)
    suggested_order: dict[str, int] = field(default_factory=dict)  # {"p": ..., "q": ...}
    ljung_box_p: float | None = None


@dataclass
class CCFResult:
    """Cross-correlation function result."""

    lags: list[int] = field(default_factory=list)
    ccf_values: list[float] = field(default_factory=list)
    confidence_bound: float = 0.0
    peak_lag: int = 0
    peak_value: float = 0.0
    significant_lags: list[int] = field(default_factory=list)
    lead_lag_interpretation: str = ""


def acf_pacf(
    data: list[float] | np.ndarray,
    n_lags: int = 20,
    ljung_box_lag: int | None = None,
) -> ACFResult:
    """Compute ACF, PACF, identify significant lags, suggest ARIMA order.

    Args:
        data: Time series values.
        n_lags: Number of lags to compute.
        ljung_box_lag: Lag for Ljung-Box test (default: min(10, n//5)).

    Returns:
        ACFResult with values, significant lags, suggested (p, q).
    """
    from statsmodels.tsa.stattools import acf, pacf

    x = np.asarray(data, dtype=float)
    x = x[np.isfinite(x)]
    n = len(x)

    n_lags = min(n_lags, n // 2 - 1)
    if n_lags < 1:
        return ACFResult()

    acf_vals = acf(x, nlags=n_lags, fft=True).tolist()
    pacf_vals = pacf(x, nlags=n_lags).tolist()

    ci = 1.96 / np.sqrt(n)

    sig_acf = [i for i in range(1, len(acf_vals)) if abs(acf_vals[i]) > ci]
    sig_pacf = [i for i in range(1, len(pacf_vals)) if abs(pacf_vals[i]) > ci]

    # ARIMA order suggestion based on cutoff patterns
    p_suggest = max(sig_pacf) if sig_pacf else 0
    q_suggest = max(sig_acf) if sig_acf else 0
    p_suggest = min(p_suggest, 5)
    q_suggest = min(q_suggest, 5)

    # Ljung-Box test
    lb_p = None
    if ljung_box_lag is None:
        ljung_box_lag = min(10, n // 5)
    if ljung_box_lag > 0:
        try:
            from statsmodels.stats.diagnostic import acorr_ljungbox
            lb = acorr_ljungbox(x, lags=[ljung_box_lag], return_df=True)
            lb_p = float(lb["lb_pvalue"].iloc[0])
        except Exception:
            pass

    return ACFResult(
        acf_values=acf_vals,
        pacf_values=pacf_vals,
        n_lags=n_lags,
        confidence_bound=float(ci),
        significant_acf_lags=sig_acf,
        significant_pacf_lags=sig_pacf,
        suggested_order={"p": p_suggest, "q": q_suggest},
        ljung_box_p=lb_p,
    )


def cross_correlation(
    x: list[float] | np.ndarray,
    y: list[float] | np.ndarray,
    max_lag: int = 20,
) -> CCFResult:
    """Cross-correlation function between two time series.

    Positive lag k means x leads y by k periods.
    Negative lag k means y leads x by |k| periods.

    Args:
        x: First time series.
        y: Second time series.
        max_lag: Maximum lag in both directions.

    Returns:
        CCFResult with CCF values, peak lag, interpretation.
    """
    x_arr = np.asarray(x, dtype=float)
    y_arr = np.asarray(y, dtype=float)
    n = min(len(x_arr), len(y_arr))

    x_std = (x_arr[:n] - np.mean(x_arr[:n]))
    y_std = (y_arr[:n] - np.mean(y_arr[:n]))
    sx = np.std(x_arr[:n])
    sy = np.std(y_arr[:n])

    if sx < 1e-15 or sy < 1e-15:
        return CCFResult()

    x_std = x_std / sx
    y_std = y_std / sy

    lags = list(range(-max_lag, max_lag + 1))
    ccf_vals = []
    for lag in lags:
        if lag >= 0:
            cc = np.mean(x_std[:n - lag] * y_std[lag:]) if lag < n else 0.0
        else:
            cc = np.mean(x_std[-lag:] * y_std[:n + lag]) if -lag < n else 0.0
        ccf_vals.append(float(cc))

    sig_bound = 2.0 / np.sqrt(n)
    sig_lags = [lag for lag, cc in zip(lags, ccf_vals) if abs(cc) > sig_bound]

    # Peak
    peak_idx = int(np.argmax(np.abs(ccf_vals)))
    peak_lag = lags[peak_idx]
    peak_val = ccf_vals[peak_idx]

    if peak_lag > 0:
        interp = f"X leads Y by {peak_lag} periods (r={peak_val:.3f})"
    elif peak_lag < 0:
        interp = f"Y leads X by {abs(peak_lag)} periods (r={peak_val:.3f})"
    else:
        interp = f"Contemporaneous correlation (r={peak_val:.3f})"

    return CCFResult(
        lags=lags,
        ccf_values=ccf_vals,
        confidence_bound=float(sig_bound),
        peak_lag=peak_lag,
        peak_value=peak_val,
        significant_lags=sig_lags,
        lead_lag_interpretation=interp,
    )
