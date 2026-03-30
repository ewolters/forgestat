"""ARIMA and SARIMA forecasting.

Requires statsmodels. Returns structured results, not statsmodels objects.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class ForecastPoint:
    """One point in a forecast."""

    step: int
    predicted: float
    ci_lower: float
    ci_upper: float


@dataclass
class ARIMAResult:
    """ARIMA/SARIMA model result."""

    order: tuple[int, int, int]  # (p, d, q)
    seasonal_order: tuple[int, int, int, int] | None = None  # (P, D, Q, m)
    aic: float = 0.0
    bic: float = 0.0
    log_likelihood: float = 0.0
    coefficients: dict[str, float] = field(default_factory=dict)
    residuals: list[float] = field(default_factory=list)
    fitted: list[float] = field(default_factory=list)
    forecast: list[ForecastPoint] = field(default_factory=list)
    ljung_box_p: float | None = None  # residual autocorrelation test
    is_stationary: bool | None = None
    adf_p_value: float | None = None


def arima(
    data: list[float] | np.ndarray,
    order: tuple[int, int, int] = (1, 1, 1),
    forecast_steps: int = 10,
    conf: float = 0.95,
) -> ARIMAResult:
    """Fit ARIMA(p,d,q) model and forecast.

    Args:
        data: Time series values.
        order: (p, d, q) — AR order, differencing, MA order.
        forecast_steps: Number of future periods to forecast.
        conf: Confidence level for forecast intervals.

    Returns:
        ARIMAResult with coefficients, diagnostics, forecasts.
    """
    from statsmodels.tsa.arima.model import ARIMA

    x = np.asarray(data, dtype=float)

    model = ARIMA(x, order=order)
    fit = model.fit()

    # Forecast
    fc = fit.get_forecast(steps=forecast_steps)
    fc_mean = fc.predicted_mean
    fc_ci = fc.conf_int(alpha=1 - conf)

    forecast_pts = [
        ForecastPoint(
            step=i + 1,
            predicted=float(fc_mean.iloc[i] if hasattr(fc_mean, 'iloc') else fc_mean[i]),
            ci_lower=float(fc_ci.iloc[i, 0] if hasattr(fc_ci, 'iloc') else fc_ci[i, 0]),
            ci_upper=float(fc_ci.iloc[i, 1] if hasattr(fc_ci, 'iloc') else fc_ci[i, 1]),
        )
        for i in range(forecast_steps)
    ]

    # Ljung-Box on residuals
    lb_p = None
    try:
        from statsmodels.stats.diagnostic import acorr_ljungbox
        lag = min(10, len(x) // 5)
        if lag > 0:
            lb = acorr_ljungbox(fit.resid, lags=[lag], return_df=True)
            lb_p = float(lb["lb_pvalue"].iloc[0])
    except Exception:
        pass

    # Stationarity
    adf_p = None
    is_stat = None
    try:
        from .stationarity import adf_test
        adf_result = adf_test(x)
        adf_p = adf_result.p_value
        is_stat = adf_result.is_stationary
    except Exception:
        pass

    # Coefficients
    coefs = {}
    for name, val in zip(fit.param_names, fit.params):
        coefs[name] = float(val)

    return ARIMAResult(
        order=order,
        aic=float(fit.aic),
        bic=float(fit.bic),
        log_likelihood=float(fit.llf),
        coefficients=coefs,
        residuals=[float(r) for r in fit.resid],
        fitted=[float(f) for f in fit.fittedvalues],
        forecast=forecast_pts,
        ljung_box_p=lb_p,
        is_stationary=is_stat,
        adf_p_value=adf_p,
    )


def sarima(
    data: list[float] | np.ndarray,
    order: tuple[int, int, int] = (1, 1, 1),
    seasonal_order: tuple[int, int, int, int] = (1, 1, 1, 12),
    forecast_steps: int = 24,
    conf: float = 0.95,
) -> ARIMAResult:
    """Fit SARIMA(p,d,q)(P,D,Q,m) model and forecast.

    Args:
        data: Time series values.
        order: (p, d, q) non-seasonal orders.
        seasonal_order: (P, D, Q, m) seasonal orders and period.
        forecast_steps: Future periods to forecast.
        conf: Confidence level.
    """
    from statsmodels.tsa.statespace.sarimax import SARIMAX

    x = np.asarray(data, dtype=float)

    model = SARIMAX(
        x, order=order, seasonal_order=seasonal_order,
        enforce_stationarity=False, enforce_invertibility=False,
    )
    fit = model.fit(disp=False, maxiter=200)

    fc = fit.get_forecast(steps=forecast_steps)
    fc_mean = fc.predicted_mean
    fc_ci = fc.conf_int(alpha=1 - conf)

    forecast_pts = [
        ForecastPoint(
            step=i + 1,
            predicted=float(fc_mean.iloc[i] if hasattr(fc_mean, 'iloc') else fc_mean[i]),
            ci_lower=float(fc_ci.iloc[i, 0] if hasattr(fc_ci, 'iloc') else fc_ci[i, 0]),
            ci_upper=float(fc_ci.iloc[i, 1] if hasattr(fc_ci, 'iloc') else fc_ci[i, 1]),
        )
        for i in range(forecast_steps)
    ]

    lb_p = None
    try:
        from statsmodels.stats.diagnostic import acorr_ljungbox
        lag = min(10, len(x) // 5)
        if lag > 0:
            lb = acorr_ljungbox(fit.resid, lags=[lag], return_df=True)
            lb_p = float(lb["lb_pvalue"].iloc[0])
    except Exception:
        pass

    coefs = {}
    for name, val in zip(fit.param_names, fit.params):
        coefs[name] = float(val)

    return ARIMAResult(
        order=order,
        seasonal_order=seasonal_order,
        aic=float(fit.aic),
        bic=float(fit.bic),
        log_likelihood=float(fit.llf),
        coefficients=coefs,
        residuals=[float(r) for r in fit.resid],
        fitted=[float(f) for f in fit.fittedvalues],
        forecast=forecast_pts,
        ljung_box_p=lb_p,
    )
