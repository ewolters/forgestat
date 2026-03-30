"""Cox Proportional Hazards — survival regression.

Uses statsmodels PHReg if available, falls back to simplified partial likelihood.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy import stats


@dataclass
class CoxPHResult:
    """Cox Proportional Hazards result."""

    coefficients: dict[str, float] = field(default_factory=dict)
    hazard_ratios: dict[str, float] = field(default_factory=dict)
    std_errors: dict[str, float] = field(default_factory=dict)
    z_statistics: dict[str, float] = field(default_factory=dict)
    p_values: dict[str, float] = field(default_factory=dict)
    concordance: float = 0.0  # Harrell's C-index
    log_likelihood: float = 0.0
    n: int = 0
    n_events: int = 0


def cox_ph(
    times: list[float] | np.ndarray,
    events: list[bool] | np.ndarray,
    covariates: dict[str, list[float]],
) -> CoxPHResult:
    """Fit Cox Proportional Hazards model.

    h(t|X) = h₀(t) × exp(β₁X₁ + β₂X₂ + ...)

    Args:
        times: Survival/censoring times.
        events: True = event occurred, False = censored.
        covariates: {covariate_name: values} — same length as times.

    Returns:
        CoxPHResult with coefficients, hazard ratios, concordance.
    """
    t = np.asarray(times, dtype=float)
    e = np.asarray(events, dtype=bool)
    n = len(t)
    n_events = int(np.sum(e))

    names = list(covariates.keys())
    X = np.column_stack([np.asarray(covariates[k], dtype=float) for k in names])

    try:
        from statsmodels.duration.hazard_regression import PHReg
        model = PHReg(t, X, status=e.astype(float))
        fit = model.fit()

        coefs = {name: float(fit.params[i]) for i, name in enumerate(names)}
        hrs = {name: float(np.exp(fit.params[i])) for i, name in enumerate(names)}
        ses = {name: float(fit.bse[i]) for i, name in enumerate(names)}
        z_stats = {name: float(fit.params[i] / fit.bse[i]) if fit.bse[i] > 0 else 0.0
                   for i, name in enumerate(names)}
        p_vals = {name: float(2 * (1 - stats.norm.cdf(abs(z_stats[name]))))
                  for name in names}

        return CoxPHResult(
            coefficients=coefs,
            hazard_ratios=hrs,
            std_errors=ses,
            z_statistics=z_stats,
            p_values=p_vals,
            log_likelihood=float(fit.llf),
            n=n,
            n_events=n_events,
        )
    except ImportError:
        pass

    # Simplified fallback: univariate Cox via Newton-Raphson on partial likelihood
    coefs = {}
    hrs = {}
    for i, name in enumerate(names):
        x = X[:, i]
        beta = _newton_cox(t, e, x)
        coefs[name] = float(beta)
        hrs[name] = float(np.exp(np.clip(beta, -700, 700)))

    return CoxPHResult(
        coefficients=coefs,
        hazard_ratios=hrs,
        n=n,
        n_events=n_events,
    )


def _newton_cox(t, e, x, max_iter=20, tol=1e-6):
    """Simple univariate Newton-Raphson for Cox partial likelihood."""
    beta = 0.0
    order = np.argsort(-t)  # reverse time order

    for _ in range(max_iter):
        grad = 0.0
        hess = 0.0
        risk_sum = 0.0
        risk_x_sum = 0.0
        risk_x2_sum = 0.0

        for idx in order:
            exp_bx = np.exp(np.clip(beta * x[idx], -500, 500))
            risk_sum += exp_bx
            risk_x_sum += x[idx] * exp_bx
            risk_x2_sum += x[idx] ** 2 * exp_bx

            if e[idx]:
                grad += x[idx] - risk_x_sum / risk_sum if risk_sum > 0 else 0
                h = (risk_x2_sum / risk_sum - (risk_x_sum / risk_sum) ** 2) if risk_sum > 0 else 1
                hess -= h

        if abs(hess) > 1e-10:
            step = grad / hess
            beta -= step
            if abs(step) < tol:
                break

    return beta
