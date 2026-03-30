"""Ordinary Least Squares regression with diagnostics.

Pure numpy/scipy — no sklearn required for core OLS.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy import stats


@dataclass
class RegressionResult:
    """OLS regression result."""

    coefficients: dict[str, float] = field(default_factory=dict)
    std_errors: dict[str, float] = field(default_factory=dict)
    t_statistics: dict[str, float] = field(default_factory=dict)
    p_values: dict[str, float] = field(default_factory=dict)
    ci_lower: dict[str, float] = field(default_factory=dict)
    ci_upper: dict[str, float] = field(default_factory=dict)
    r_squared: float = 0.0
    adj_r_squared: float = 0.0
    f_statistic: float = 0.0
    f_p_value: float = 0.0
    durbin_watson: float = 0.0
    mse: float = 0.0
    rmse: float = 0.0
    n: int = 0
    p: int = 0
    residuals: list[float] = field(default_factory=list)
    fitted: list[float] = field(default_factory=list)
    leverage: list[float] = field(default_factory=list)
    cooks_distance: list[float] = field(default_factory=list)


def ols(
    X: list[list[float]] | np.ndarray,
    y: list[float] | np.ndarray,
    feature_names: list[str] | None = None,
    alpha: float = 0.05,
    add_intercept: bool = True,
) -> RegressionResult:
    """Ordinary Least Squares regression.

    Args:
        X: Predictor matrix (n x p).
        y: Response vector (n,).
        feature_names: Names for predictors.
        alpha: Significance level for CIs.
        add_intercept: If True, prepend a column of ones.

    Returns:
        RegressionResult with coefficients, diagnostics, and fit metrics.
    """
    X_arr = np.asarray(X, dtype=float)
    y_arr = np.asarray(y, dtype=float).ravel()
    n = len(y_arr)

    if X_arr.ndim == 1:
        X_arr = X_arr.reshape(-1, 1)

    p_orig = X_arr.shape[1]
    names = feature_names or [f"X{i+1}" for i in range(p_orig)]

    if add_intercept:
        X_full = np.column_stack([np.ones(n), X_arr])
        all_names = ["Intercept"] + list(names)
    else:
        X_full = X_arr
        all_names = list(names)

    p = X_full.shape[1]

    # Solve normal equations
    try:
        XtX_inv = np.linalg.inv(X_full.T @ X_full)
    except np.linalg.LinAlgError:
        XtX_inv = np.linalg.pinv(X_full.T @ X_full)

    beta = XtX_inv @ X_full.T @ y_arr
    y_pred = X_full @ beta
    residuals = y_arr - y_pred

    # Sum of squares
    ss_res = float(np.sum(residuals ** 2))
    ss_tot = float(np.sum((y_arr - np.mean(y_arr)) ** 2))
    df_res = n - p
    df_reg = p - (1 if add_intercept else 0)

    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0
    adj_r2 = 1 - (1 - r2) * (n - 1) / df_res if df_res > 0 else 0.0
    mse = ss_res / df_res if df_res > 0 else 0.0
    rmse = float(np.sqrt(mse))

    # F-statistic
    ss_reg = ss_tot - ss_res
    ms_reg = ss_reg / df_reg if df_reg > 0 else 0.0
    f_stat = ms_reg / mse if mse > 0 else 0.0
    f_p = float(1 - stats.f.cdf(f_stat, df_reg, df_res)) if df_reg > 0 and df_res > 0 else 1.0

    # Coefficient standard errors, t-stats, p-values, CIs
    var_beta = mse * np.diag(XtX_inv)
    se = np.sqrt(np.maximum(var_beta, 0))
    t_stats = np.where(se > 0, beta / se, 0.0)
    p_vals = 2 * (1 - stats.t.cdf(np.abs(t_stats), df_res)) if df_res > 0 else np.ones_like(beta)
    t_crit = stats.t.ppf(1 - alpha / 2, df_res) if df_res > 0 else 0

    # Durbin-Watson
    dw = float(np.sum(np.diff(residuals) ** 2) / ss_res) if ss_res > 0 else 2.0

    # Hat matrix and Cook's distance
    H = X_full @ XtX_inv @ X_full.T
    h = np.diag(H)
    std_res = residuals / np.sqrt(mse * np.maximum(1 - h, 1e-10)) if mse > 0 else np.zeros(n)
    cooks = (std_res ** 2 * h) / (p * (1 - h) ** 2) if p > 0 else np.zeros(n)

    return RegressionResult(
        coefficients={name: float(b) for name, b in zip(all_names, beta)},
        std_errors={name: float(s) for name, s in zip(all_names, se)},
        t_statistics={name: float(t) for name, t in zip(all_names, t_stats)},
        p_values={name: float(pv) for name, pv in zip(all_names, p_vals)},
        ci_lower={name: float(b - t_crit * s) for name, b, s in zip(all_names, beta, se)},
        ci_upper={name: float(b + t_crit * s) for name, b, s in zip(all_names, beta, se)},
        r_squared=r2,
        adj_r_squared=adj_r2,
        f_statistic=float(f_stat),
        f_p_value=f_p,
        durbin_watson=dw,
        mse=mse,
        rmse=rmse,
        n=n,
        p=df_reg,
        residuals=residuals.tolist(),
        fitted=y_pred.tolist(),
        leverage=h.tolist(),
        cooks_distance=cooks.tolist(),
    )


def polynomial(
    x: list[float] | np.ndarray,
    y: list[float] | np.ndarray,
    degree: int = 2,
    alpha: float = 0.05,
) -> RegressionResult:
    """Polynomial regression: y = β₀ + β₁x + β₂x² + ... + βₖxᵏ.

    Args:
        x: Single predictor (n,).
        y: Response vector (n,).
        degree: Polynomial degree.
        alpha: Significance level.
    """
    x_arr = np.asarray(x, dtype=float).ravel()
    X_poly = np.column_stack([x_arr ** d for d in range(1, degree + 1)])
    names = [f"x^{d}" if d > 1 else "x" for d in range(1, degree + 1)]
    return ols(X_poly, y, feature_names=names, alpha=alpha)
