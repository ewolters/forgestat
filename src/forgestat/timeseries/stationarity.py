"""Stationarity tests and differencing — ADF, KPSS, auto-differencing.

Foundation for ARIMA order selection.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class StationarityResult:
    """Stationarity test result."""

    test_name: str
    statistic: float
    p_value: float
    is_stationary: bool
    n_diffs_needed: int = 0
    critical_values: dict[str, float] | None = None
    detail: str = ""


def adf_test(
    data: list[float] | np.ndarray,
    alpha: float = 0.05,
) -> StationarityResult:
    """Augmented Dickey-Fuller test for unit root.

    H₀: Series has a unit root (non-stationary).
    H₁: Series is stationary.

    Args:
        data: Time series values.
        alpha: Significance level.

    Returns:
        StationarityResult — stationary if p < alpha.
    """
    from statsmodels.tsa.stattools import adfuller

    x = np.asarray(data, dtype=float)
    x = x[np.isfinite(x)]

    result = adfuller(x, autolag="AIC")
    stat, p, used_lag, nobs, crit, icbest = result

    return StationarityResult(
        test_name="Augmented Dickey-Fuller",
        statistic=float(stat),
        p_value=float(p),
        is_stationary=bool(p < alpha),
        critical_values={k: float(v) for k, v in crit.items()},
        detail=f"ADF={stat:.4f}, p={p:.4f}, lags={used_lag}",
    )


def kpss_test(
    data: list[float] | np.ndarray,
    regression: str = "c",
    alpha: float = 0.05,
) -> StationarityResult:
    """KPSS test for stationarity.

    H₀: Series is stationary.
    H₁: Series has a unit root (non-stationary).

    Note: Opposite null hypothesis from ADF — use both for confirmation.

    Args:
        data: Time series values.
        regression: "c" (level stationarity) or "ct" (trend stationarity).
        alpha: Significance level.
    """
    from statsmodels.tsa.stattools import kpss

    x = np.asarray(data, dtype=float)
    x = x[np.isfinite(x)]

    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        stat, p, lags, crit = kpss(x, regression=regression, nlags="auto")

    return StationarityResult(
        test_name="KPSS",
        statistic=float(stat),
        p_value=float(p),
        is_stationary=bool(p >= alpha),  # KPSS: reject = non-stationary
        critical_values={k: float(v) for k, v in crit.items()},
        detail=f"KPSS={stat:.4f}, p={p:.4f}, lags={lags}",
    )


def auto_diff_order(
    data: list[float] | np.ndarray,
    max_d: int = 2,
    alpha: float = 0.05,
) -> int:
    """Determine number of differences needed for stationarity.

    Repeatedly applies ADF test after differencing.

    Args:
        data: Time series values.
        max_d: Maximum differencing order.
        alpha: Significance level for ADF test.

    Returns:
        Recommended differencing order d.
    """
    x = np.asarray(data, dtype=float)
    x = x[np.isfinite(x)]

    for d in range(max_d + 1):
        result = adf_test(x)
        if result.is_stationary:
            return d
        x = np.diff(x)
        if len(x) < 10:
            return d

    return max_d
