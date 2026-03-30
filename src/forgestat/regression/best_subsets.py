"""Best subsets regression — exhaustive search over predictor combinations.

Evaluates all 2^p - 1 subsets (up to max_predictors) and ranks by AIC/BIC/Adj-R².
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from itertools import combinations

import numpy as np

from .linear import RegressionResult, ols


@dataclass
class SubsetScore:
    """Score for one predictor subset."""

    features: list[str]
    p: int
    r_squared: float = 0.0
    adj_r_squared: float = 0.0
    aic: float = 0.0
    bic: float = 0.0
    mallows_cp: float = 0.0


@dataclass
class BestSubsetsResult:
    """Best subsets regression result."""

    best_aic: SubsetScore | None = None
    best_bic: SubsetScore | None = None
    best_adj_r2: SubsetScore | None = None
    all_subsets: list[SubsetScore] = field(default_factory=list)
    full_model: RegressionResult | None = None


def best_subsets(
    X: list[list[float]] | np.ndarray,
    y: list[float] | np.ndarray,
    feature_names: list[str] | None = None,
    max_predictors: int | None = None,
) -> BestSubsetsResult:
    """Exhaustive best subsets regression.

    Args:
        X: Full predictor matrix (n × p).
        y: Response vector.
        feature_names: Names for all predictors.
        max_predictors: Max subset size (default: min(p, 15) to avoid combinatorial explosion).

    Returns:
        BestSubsetsResult with best model by AIC, BIC, and adjusted R².
    """
    X_arr = np.asarray(X, dtype=float)
    y_arr = np.asarray(y, dtype=float).ravel()
    n = len(y_arr)

    if X_arr.ndim == 1:
        X_arr = X_arr.reshape(-1, 1)

    p = X_arr.shape[1]
    names = feature_names or [f"X{i+1}" for i in range(p)]
    max_p = max_predictors or min(p, 15)

    # Full model MSE for Mallows' Cp
    full_result = ols(X_arr, y_arr, feature_names=names)
    full_mse = full_result.mse

    all_subsets = []

    for k in range(1, max_p + 1):
        for combo in combinations(range(p), k):
            X_sub = X_arr[:, list(combo)]
            sub_names = [names[i] for i in combo]
            result = ols(X_sub, y_arr, feature_names=sub_names)

            ss_res = result.mse * (n - k - 1) if n > k + 1 else 0
            aic = n * math.log(max(ss_res / n, 1e-15)) + 2 * (k + 1)
            bic = n * math.log(max(ss_res / n, 1e-15)) + (k + 1) * math.log(n)
            cp = ss_res / full_mse - n + 2 * (k + 1) if full_mse > 0 else float("inf")

            all_subsets.append(SubsetScore(
                features=sub_names,
                p=k,
                r_squared=result.r_squared,
                adj_r_squared=result.adj_r_squared,
                aic=aic,
                bic=bic,
                mallows_cp=cp,
            ))

    best_aic = min(all_subsets, key=lambda s: s.aic) if all_subsets else None
    best_bic = min(all_subsets, key=lambda s: s.bic) if all_subsets else None
    best_adj = max(all_subsets, key=lambda s: s.adj_r_squared) if all_subsets else None

    return BestSubsetsResult(
        best_aic=best_aic,
        best_bic=best_bic,
        best_adj_r2=best_adj,
        all_subsets=all_subsets,
        full_model=full_result,
    )
