"""Robust regression — outlier-resistant estimation.

Huber M-estimator (iteratively reweighted least squares).
Pure numpy/scipy — statsmodels optional for advanced methods.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class RobustRegressionResult:
    """Robust regression result with OLS comparison."""

    method: str  # "huber", "bisquare"
    coefficients: dict[str, float] = field(default_factory=dict)
    ols_coefficients: dict[str, float] = field(default_factory=dict)
    coefficient_changes: dict[str, float] = field(default_factory=dict)  # pct change from OLS
    weights: list[float] = field(default_factory=list)  # per-observation weights
    n_downweighted: int = 0  # observations with weight < 0.5
    r_squared: float = 0.0
    residuals: list[float] = field(default_factory=list)


def robust_regression(
    X: list[list[float]] | np.ndarray,
    y: list[float] | np.ndarray,
    feature_names: list[str] | None = None,
    method: str = "huber",
    max_iter: int = 50,
    tol: float = 1e-6,
    huber_c: float = 1.345,
) -> RobustRegressionResult:
    """Robust regression via iteratively reweighted least squares (IRLS).

    M-estimators downweight outliers. Huber uses linear penalty beyond c·σ.
    Bisquare (Tukey) gives zero weight beyond 4.685·σ.

    Args:
        X: Predictor matrix (n × p).
        y: Response vector.
        feature_names: Predictor names.
        method: "huber" or "bisquare".
        max_iter: Maximum IRLS iterations.
        tol: Convergence tolerance.
        huber_c: Tuning constant for Huber (default 1.345 = 95% efficiency).

    Returns:
        RobustRegressionResult with robust + OLS coefficients for comparison.
    """
    X_arr = np.asarray(X, dtype=float)
    y_arr = np.asarray(y, dtype=float).ravel()
    n = len(y_arr)

    if X_arr.ndim == 1:
        X_arr = X_arr.reshape(-1, 1)

    p_orig = X_arr.shape[1]
    names = feature_names or [f"X{i+1}" for i in range(p_orig)]
    all_names = ["Intercept"] + list(names)

    # Add intercept
    X_full = np.column_stack([np.ones(n), X_arr])
    p = X_full.shape[1]

    # OLS for comparison
    try:
        ols_beta = np.linalg.lstsq(X_full, y_arr, rcond=None)[0]
    except np.linalg.LinAlgError:
        ols_beta = np.zeros(p)

    # IRLS
    beta = ols_beta.copy()
    weights = np.ones(n)

    if method == "bisquare":
        c_val = 4.685
    else:  # huber
        c_val = huber_c

    for _ in range(max_iter):
        residuals = y_arr - X_full @ beta
        mad = np.median(np.abs(residuals - np.median(residuals)))
        sigma = mad / 0.6745 if mad > 0 else 1.0

        u = residuals / sigma

        # Weight function
        if method == "bisquare":
            weights = np.where(np.abs(u) <= c_val, (1 - (u / c_val) ** 2) ** 2, 0.0)
        else:  # huber
            weights = np.where(np.abs(u) <= c_val, 1.0, c_val / np.abs(u))

        weights = np.maximum(weights, 1e-10)

        # Weighted least squares
        W = np.diag(weights)
        try:
            beta_new = np.linalg.solve(X_full.T @ W @ X_full, X_full.T @ W @ y_arr)
        except np.linalg.LinAlgError:
            break

        if np.max(np.abs(beta_new - beta)) < tol:
            beta = beta_new
            break
        beta = beta_new

    # Final residuals and R²
    y_pred = X_full @ beta
    residuals = y_arr - y_pred
    ss_res = float(np.sum(residuals ** 2))
    ss_tot = float(np.sum((y_arr - np.mean(y_arr)) ** 2))
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0

    # Coefficient changes from OLS
    changes = {}
    for i, name in enumerate(all_names):
        if abs(ols_beta[i]) > 1e-10:
            changes[name] = float((beta[i] - ols_beta[i]) / ols_beta[i] * 100)
        else:
            changes[name] = 0.0

    n_downweighted = int(np.sum(weights < 0.5))

    return RobustRegressionResult(
        method=method,
        coefficients={name: float(b) for name, b in zip(all_names, beta)},
        ols_coefficients={name: float(b) for name, b in zip(all_names, ols_beta)},
        coefficient_changes=changes,
        weights=weights.tolist(),
        n_downweighted=n_downweighted,
        r_squared=r2,
        residuals=residuals.tolist(),
    )
