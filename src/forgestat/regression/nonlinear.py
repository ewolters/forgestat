"""Nonlinear regression — curve fitting with preset and custom models.

Uses scipy.optimize.curve_fit (Levenberg-Marquardt).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
from scipy import optimize


@dataclass
class NonlinearResult:
    """Nonlinear curve fit result."""

    model: str
    parameters: dict[str, float] = field(default_factory=dict)
    std_errors: dict[str, float] = field(default_factory=dict)
    r_squared: float = 0.0
    rmse: float = 0.0
    aic: float = 0.0
    bic: float = 0.0
    n: int = 0
    converged: bool = True
    residuals: list[float] = field(default_factory=list)
    fitted: list[float] = field(default_factory=list)


# Preset model functions and their parameter names
_MODELS = {
    "exponential": (lambda x, a, b: a * np.exp(b * x), ["a", "b"]),
    "power": (lambda x, a, b: a * np.maximum(x, 1e-10) ** b, ["a", "b"]),
    "logarithmic": (lambda x, a, b: a * np.log(np.maximum(x, 1e-10)) + b, ["a", "b"]),
    "logistic": (lambda x, L, k, x0: L / (1 + np.exp(-k * (x - x0))), ["L", "k", "x0"]),
    "michaelis_menten": (lambda x, Vmax, Km: Vmax * x / (Km + x), ["Vmax", "Km"]),
    "gompertz": (lambda x, a, b, c: a * np.exp(-b * np.exp(-c * x)), ["a", "b", "c"]),
    "hill": (lambda x, Vmax, Kd, n: Vmax * x ** n / (Kd ** n + x ** n), ["Vmax", "Kd", "n"]),
}


def curve_fit(
    x: list[float] | np.ndarray,
    y: list[float] | np.ndarray,
    model: str = "exponential",
    p0: list[float] | None = None,
    maxfev: int = 10000,
) -> NonlinearResult:
    """Fit a nonlinear model to data.

    Args:
        x: Independent variable values.
        y: Dependent variable values.
        model: One of the preset model names, or "polynomial2"/"polynomial3".
        p0: Initial parameter guesses (auto-estimated if None).
        maxfev: Maximum function evaluations.

    Returns:
        NonlinearResult with parameters, fit metrics, diagnostics.
    """
    x_arr = np.asarray(x, dtype=float).ravel()
    y_arr = np.asarray(y, dtype=float).ravel()
    n = len(x_arr)

    if model in ("polynomial2", "polynomial3"):
        deg = 2 if model == "polynomial2" else 3
        coeffs = np.polyfit(x_arr, y_arr, deg)
        y_pred = np.polyval(coeffs, x_arr)
        param_names = [f"a{i}" for i in range(deg + 1)]
        popt = coeffs
        perr = np.zeros(len(coeffs))
    elif model in _MODELS:
        func, param_names = _MODELS[model]
        try:
            popt, pcov = optimize.curve_fit(func, x_arr, y_arr, p0=p0, maxfev=maxfev)
            perr = np.sqrt(np.maximum(np.diag(pcov), 0))
            y_pred = func(x_arr, *popt)
        except RuntimeError:
            return NonlinearResult(model=model, converged=False, n=n)
    else:
        raise ValueError(f"Unknown model: {model}. Available: {list(_MODELS.keys())}")

    residuals = y_arr - y_pred
    ss_res = float(np.sum(residuals ** 2))
    ss_tot = float(np.sum((y_arr - np.mean(y_arr)) ** 2))
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0
    rmse = math.sqrt(ss_res / n) if n > 0 else 0.0

    k = len(popt)
    aic = n * math.log(ss_res / n) + 2 * k if n > 0 and ss_res > 0 else float("inf")
    bic = n * math.log(ss_res / n) + k * math.log(n) if n > 0 and ss_res > 0 else float("inf")

    return NonlinearResult(
        model=model,
        parameters={name: float(v) for name, v in zip(param_names, popt)},
        std_errors={name: float(e) for name, e in zip(param_names, perr)},
        r_squared=r2,
        rmse=rmse,
        aic=aic,
        bic=bic,
        n=n,
        converged=True,
        residuals=residuals.tolist(),
        fitted=y_pred.tolist(),
    )
