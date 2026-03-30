"""Logistic and Poisson regression — GLM family.

Core logistic uses numpy/scipy only. Poisson uses statsmodels if available.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy import stats


@dataclass
class LogisticResult:
    """Binary logistic regression result."""

    coefficients: dict[str, float] = field(default_factory=dict)
    odds_ratios: dict[str, float] = field(default_factory=dict)
    std_errors: dict[str, float] = field(default_factory=dict)
    z_statistics: dict[str, float] = field(default_factory=dict)
    p_values: dict[str, float] = field(default_factory=dict)
    log_likelihood: float = 0.0
    null_log_likelihood: float = 0.0
    pseudo_r_squared: float = 0.0  # McFadden's
    aic: float = 0.0
    n: int = 0
    converged: bool = True


@dataclass
class PoissonResult:
    """Poisson regression result."""

    coefficients: dict[str, float] = field(default_factory=dict)
    irr: dict[str, float] = field(default_factory=dict)  # incidence rate ratios
    std_errors: dict[str, float] = field(default_factory=dict)
    z_statistics: dict[str, float] = field(default_factory=dict)
    p_values: dict[str, float] = field(default_factory=dict)
    deviance: float = 0.0
    pearson_chi2: float = 0.0
    aic: float = 0.0
    n: int = 0


def _sigmoid(z):
    return 1 / (1 + np.exp(-np.clip(z, -500, 500)))


def logistic_regression(
    X: list[list[float]] | np.ndarray,
    y: list[int] | np.ndarray,
    feature_names: list[str] | None = None,
    max_iter: int = 100,
    tol: float = 1e-8,
) -> LogisticResult:
    """Binary logistic regression via iteratively reweighted least squares (IRLS).

    Args:
        X: Predictor matrix (n x p).
        y: Binary response (0/1).
        feature_names: Names for predictors.
        max_iter: Maximum IRLS iterations.
        tol: Convergence tolerance.

    Returns:
        LogisticResult with coefficients, odds ratios, fit metrics.
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

    # IRLS
    beta = np.zeros(p)
    converged = False

    for _ in range(max_iter):
        eta = X_full @ beta
        mu = _sigmoid(eta)
        mu = np.clip(mu, 1e-10, 1 - 1e-10)

        W = np.diag(mu * (1 - mu))
        z = eta + (y_arr - mu) / (mu * (1 - mu))

        try:
            XtWX = X_full.T @ W @ X_full
            XtWz = X_full.T @ W @ z
            beta_new = np.linalg.solve(XtWX, XtWz)
        except np.linalg.LinAlgError:
            break

        if np.max(np.abs(beta_new - beta)) < tol:
            beta = beta_new
            converged = True
            break
        beta = beta_new

    # Final predictions
    mu = _sigmoid(X_full @ beta)
    mu = np.clip(mu, 1e-10, 1 - 1e-10)

    # Log-likelihood
    ll = float(np.sum(y_arr * np.log(mu) + (1 - y_arr) * np.log(1 - mu)))
    p_bar = np.mean(y_arr)
    ll_null = float(n * (p_bar * np.log(p_bar + 1e-10) + (1 - p_bar) * np.log(1 - p_bar + 1e-10)))
    pseudo_r2 = 1 - ll / ll_null if ll_null != 0 else 0.0

    # Standard errors from Fisher information
    W_final = np.diag(mu * (1 - mu))
    try:
        cov = np.linalg.inv(X_full.T @ W_final @ X_full)
        se = np.sqrt(np.maximum(np.diag(cov), 0))
    except np.linalg.LinAlgError:
        se = np.zeros(p)

    z_stats = np.where(se > 0, beta / se, 0.0)
    p_vals = 2 * (1 - stats.norm.cdf(np.abs(z_stats)))
    odds_ratios = np.exp(np.clip(beta, -700, 700))

    aic = -2 * ll + 2 * p

    return LogisticResult(
        coefficients={name: float(b) for name, b in zip(all_names, beta)},
        odds_ratios={name: float(o) for name, o in zip(all_names, odds_ratios)},
        std_errors={name: float(s) for name, s in zip(all_names, se)},
        z_statistics={name: float(z) for name, z in zip(all_names, z_stats)},
        p_values={name: float(pv) for name, pv in zip(all_names, p_vals)},
        log_likelihood=ll,
        null_log_likelihood=ll_null,
        pseudo_r_squared=pseudo_r2,
        aic=aic,
        n=n,
        converged=converged,
    )


def poisson_regression(
    X: list[list[float]] | np.ndarray,
    y: list[int] | np.ndarray,
    feature_names: list[str] | None = None,
) -> PoissonResult:
    """Poisson regression for count data.

    Uses statsmodels GLM if available, falls back to IRLS.

    Args:
        X: Predictor matrix.
        y: Count response (non-negative integers).
        feature_names: Predictor names.
    """
    X_arr = np.asarray(X, dtype=float)
    y_arr = np.asarray(y, dtype=float).ravel()
    n = len(y_arr)

    if X_arr.ndim == 1:
        X_arr = X_arr.reshape(-1, 1)

    p_orig = X_arr.shape[1]
    names = feature_names or [f"X{i+1}" for i in range(p_orig)]
    all_names = ["Intercept"] + list(names)

    X_full = np.column_stack([np.ones(n), X_arr])
    p = X_full.shape[1]

    try:
        import statsmodels.api as sm
        model = sm.GLM(y_arr, X_full, family=sm.families.Poisson())
        fit = model.fit()

        beta = fit.params
        se = fit.bse
        z_stats = beta / np.where(se > 0, se, 1e-10)
        p_vals = 2 * (1 - stats.norm.cdf(np.abs(z_stats)))

        return PoissonResult(
            coefficients={name: float(b) for name, b in zip(all_names, beta)},
            irr={name: float(np.exp(b)) for name, b in zip(all_names, beta)},
            std_errors={name: float(s) for name, s in zip(all_names, se)},
            z_statistics={name: float(z) for name, z in zip(all_names, z_stats)},
            p_values={name: float(pv) for name, pv in zip(all_names, p_vals)},
            deviance=float(fit.deviance),
            pearson_chi2=float(fit.pearson_chi2),
            aic=float(fit.aic),
            n=n,
        )
    except ImportError:
        pass

    # Fallback: IRLS for Poisson
    beta = np.zeros(p)
    for _ in range(50):
        eta = X_full @ beta
        mu = np.exp(np.clip(eta, -20, 20))
        mu = np.maximum(mu, 1e-10)
        W = np.diag(mu)
        z = eta + (y_arr - mu) / mu
        try:
            beta_new = np.linalg.solve(X_full.T @ W @ X_full, X_full.T @ W @ z)
        except np.linalg.LinAlgError:
            break
        if np.max(np.abs(beta_new - beta)) < 1e-8:
            beta = beta_new
            break
        beta = beta_new

    mu = np.exp(np.clip(X_full @ beta, -20, 20))
    deviance = 2 * float(np.sum(y_arr * np.log(np.maximum(y_arr, 1e-10) / mu) - (y_arr - mu)))
    ll = float(np.sum(y_arr * np.log(mu) - mu))

    return PoissonResult(
        coefficients={name: float(b) for name, b in zip(all_names, beta)},
        irr={name: float(np.exp(b)) for name, b in zip(all_names, beta)},
        std_errors={},
        z_statistics={},
        p_values={},
        deviance=deviance,
        pearson_chi2=float(np.sum((y_arr - mu) ** 2 / mu)),
        aic=-2 * ll + 2 * p,
        n=n,
    )
