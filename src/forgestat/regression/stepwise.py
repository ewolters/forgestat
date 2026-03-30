"""Stepwise regression — forward, backward, and bidirectional selection.

Uses OLS from linear.py. Requires statsmodels for p-value computation.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .linear import RegressionResult, ols


@dataclass
class StepwiseResult:
    """Stepwise regression result."""

    final_model: RegressionResult | None = None
    selected_features: list[str] = field(default_factory=list)
    steps: list[dict] = field(default_factory=list)  # [{action, variable, p_value, r_squared}]
    method: str = "both"


def stepwise(
    X: list[list[float]] | np.ndarray,
    y: list[float] | np.ndarray,
    feature_names: list[str] | None = None,
    method: str = "both",
    alpha_enter: float = 0.05,
    alpha_remove: float = 0.10,
    max_steps: int = 100,
) -> StepwiseResult:
    """Stepwise feature selection with OLS regression.

    Args:
        X: Full predictor matrix (n x p).
        y: Response vector.
        feature_names: Names for all predictors.
        method: "forward", "backward", or "both".
        alpha_enter: p-value threshold to enter (forward step).
        alpha_remove: p-value threshold to remove (backward step).
        max_steps: Maximum selection steps.

    Returns:
        StepwiseResult with selected features, step history, and final OLS model.
    """
    X_arr = np.asarray(X, dtype=float)
    y_arr = np.asarray(y, dtype=float).ravel()

    if X_arr.ndim == 1:
        X_arr = X_arr.reshape(-1, 1)

    p = X_arr.shape[1]
    names = feature_names or [f"X{i+1}" for i in range(p)]

    if method == "backward":
        selected = list(range(p))
    else:
        selected = []

    remaining = list(set(range(p)) - set(selected))
    steps = []

    for _ in range(max_steps):
        changed = False

        # Forward step
        if method in ("forward", "both") and remaining:
            best_p = 1.0
            best_idx = None
            for idx in remaining:
                candidate = selected + [idx]
                result = ols(X_arr[:, candidate], y_arr, feature_names=[names[i] for i in candidate])
                # Get p-value of the new variable (last non-intercept)
                var_name = names[idx]
                pv = result.p_values.get(var_name, 1.0)
                if pv < best_p:
                    best_p = pv
                    best_idx = idx

            if best_idx is not None and best_p < alpha_enter:
                selected.append(best_idx)
                remaining.remove(best_idx)
                result = ols(X_arr[:, selected], y_arr, feature_names=[names[i] for i in selected])
                steps.append({
                    "action": "enter",
                    "variable": names[best_idx],
                    "p_value": best_p,
                    "r_squared": result.r_squared,
                })
                changed = True

        # Backward step
        if method in ("backward", "both") and len(selected) > 1:
            worst_p = 0.0
            worst_idx = None
            result = ols(X_arr[:, selected], y_arr, feature_names=[names[i] for i in selected])
            for idx in selected:
                var_name = names[idx]
                pv = result.p_values.get(var_name, 0.0)
                if pv > worst_p:
                    worst_p = pv
                    worst_idx = idx

            if worst_idx is not None and worst_p > alpha_remove:
                selected.remove(worst_idx)
                remaining.append(worst_idx)
                result = ols(X_arr[:, selected], y_arr, feature_names=[names[i] for i in selected])
                steps.append({
                    "action": "remove",
                    "variable": names[worst_idx],
                    "p_value": worst_p,
                    "r_squared": result.r_squared,
                })
                changed = True

        if not changed:
            break

    # Final model
    if selected:
        final = ols(X_arr[:, selected], y_arr, feature_names=[names[i] for i in selected])
    else:
        final = None

    return StepwiseResult(
        final_model=final,
        selected_features=[names[i] for i in selected],
        steps=steps,
        method=method,
    )
