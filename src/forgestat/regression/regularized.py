"""Regularized regression — Ridge, Lasso, Elastic Net.

Pure numpy/scipy — no sklearn required.
Essential for high-dimensional process data where OLS fails due to multicollinearity.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class RegularizedResult:
    """Regularized regression result."""

    method: str  # "ridge", "lasso", "elastic_net"
    coefficients: dict[str, float] = field(default_factory=dict)
    intercept: float = 0.0
    alpha: float = 0.0
    l1_ratio: float | None = None
    r_squared: float = 0.0
    r_squared_adj: float = 0.0
    mse: float = 0.0
    n_features_selected: int = 0
    selected_features: list[str] = field(default_factory=list)
    cv_score: float | None = None
    standardized_coef: dict[str, float] = field(default_factory=dict)
    residuals: list[float] = field(default_factory=list)


@dataclass
class PathResult:
    """Coefficient path across regularization values."""

    alphas: list[float] = field(default_factory=list)
    coefficients: dict[str, list[float]] = field(default_factory=dict)
    best_alpha: float = 0.0
    method: str = ""


def _standardize(X: np.ndarray):
    """Center and scale columns. Returns (X_std, means, stds)."""
    means = X.mean(axis=0)
    stds = X.std(axis=0, ddof=0)
    stds[stds == 0] = 1.0  # prevent division by zero for constant columns
    return (X - means) / stds, means, stds


def _r_squared(y: np.ndarray, y_pred: np.ndarray) -> float:
    ss_res = np.sum((y - y_pred) ** 2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)
    return float(1 - ss_res / ss_tot) if ss_tot > 0 else 0.0


def _adj_r_squared(r2: float, n: int, p: int) -> float:
    denom = n - p - 1
    return float(1 - (1 - r2) * (n - 1) / denom) if denom > 0 else 0.0


def _soft_threshold(x: float, lam: float) -> float:
    """Soft-thresholding operator for Lasso coordinate descent."""
    if x > lam:
        return x - lam
    elif x < -lam:
        return x + lam
    return 0.0


def _coordinate_descent(
    X: np.ndarray,
    y: np.ndarray,
    alpha: float,
    l1_ratio: float = 1.0,
    max_iter: int = 1000,
    tol: float = 1e-6,
) -> np.ndarray:
    """Coordinate descent for Elastic Net (Lasso when l1_ratio=1).

    Solves: min (1/2n)||y - Xβ||² + α * l1_ratio * ||β||₁ + α * (1-l1_ratio)/2 * ||β||²₂

    X should be standardized (zero mean, unit variance columns).
    y should be centered.
    """
    n, p = X.shape
    beta = np.zeros(p)
    l1_pen = alpha * l1_ratio
    l2_pen = alpha * (1 - l1_ratio)

    # Precompute X'X diagonal and X'y
    # For coordinate descent, we need column norms
    col_norms_sq = np.sum(X ** 2, axis=0)  # ||X_j||^2

    for _ in range(max_iter):
        beta_old = beta.copy()
        for j in range(p):
            # Partial residual excluding feature j
            r_j = y - X @ beta + X[:, j] * beta[j]
            # Unnormalized update
            rho = X[:, j] @ r_j / n
            # Apply elastic net penalty
            denom = col_norms_sq[j] / n + l2_pen
            if denom == 0:
                beta[j] = 0.0
            else:
                beta[j] = _soft_threshold(rho, l1_pen) / denom
        if np.max(np.abs(beta - beta_old)) < tol:
            break

    return beta


def _kfold_indices(n: int, k: int = 5, seed: int = 0):
    """Generate k-fold cross-validation index splits."""
    rng = np.random.RandomState(seed)
    indices = rng.permutation(n)
    fold_size = n // k
    for i in range(k):
        start = i * fold_size
        end = start + fold_size if i < k - 1 else n
        test_idx = indices[start:end]
        train_idx = np.concatenate([indices[:start], indices[end:]])
        yield train_idx, test_idx


def _gcv_ridge(X: np.ndarray, y: np.ndarray, alphas: np.ndarray) -> float:
    """Generalized Cross-Validation for Ridge to pick best alpha."""
    n = X.shape[0]
    # SVD of X
    U, s, Vt = np.linalg.svd(X, full_matrices=False)

    best_alpha = alphas[0]
    best_gcv = np.inf

    for a in alphas:
        # Ridge predictions via SVD: y_hat = U diag(s^2/(s^2+a)) U' y
        d = s ** 2 / (s ** 2 + a)
        y_hat = U @ (d[:, None] * (U.T @ y[:, None])).ravel()
        residuals = y - y_hat
        # Effective df = sum(d)
        df_eff = np.sum(d)
        gcv = np.sum(residuals ** 2) / (n * (1 - df_eff / n) ** 2) if df_eff < n else np.inf
        if gcv < best_gcv:
            best_gcv = gcv
            best_alpha = a
    return float(best_alpha)


def ridge(
    X: list[list[float]] | np.ndarray,
    y: list[float] | np.ndarray,
    alpha: float | None = 1.0,
    feature_names: list[str] | None = None,
    conf: float = 0.95,
    alphas: list[float] | np.ndarray | None = None,
) -> RegularizedResult:
    """Ridge regression (L2 penalty).

    Args:
        X: Predictor matrix (n x p).
        y: Response vector (n,).
        alpha: Regularization strength. Ignored if alphas provided.
        feature_names: Names for predictors.
        conf: Confidence level (reserved for future CI support).
        alphas: If provided, cross-validate to pick best alpha.
            If both alpha and alphas are None, auto-select via GCV.
    """
    X_arr = np.asarray(X, dtype=float)
    y_arr = np.asarray(y, dtype=float).ravel()
    n = len(y_arr)

    if X_arr.ndim == 1:
        X_arr = X_arr.reshape(-1, 1)

    p = X_arr.shape[1]
    names = feature_names or [f"X{i+1}" for i in range(p)]

    # Standardize X, center y
    X_std, X_means, X_stds = _standardize(X_arr)
    y_mean = float(np.mean(y_arr))
    y_c = y_arr - y_mean

    # Select alpha
    cv_score = None
    if alphas is not None:
        alphas_arr = np.asarray(alphas, dtype=float)
        best_alpha = _gcv_ridge(X_std, y_c, alphas_arr)
        # Compute CV score for the best alpha
        cv_score = _ridge_cv_score(X_std, y_c, best_alpha)
        alpha = best_alpha
    elif alpha is None:
        # Auto-select via GCV with default range
        alphas_arr = np.logspace(-4, 4, 50)
        best_alpha = _gcv_ridge(X_std, y_c, alphas_arr)
        cv_score = _ridge_cv_score(X_std, y_c, best_alpha)
        alpha = best_alpha

    # Fit: β_std = (X'X + αI)⁻¹ X'y on standardized data
    XtX = X_std.T @ X_std
    beta_std = np.linalg.solve(XtX + alpha * np.eye(p), X_std.T @ y_c)

    # Unstandardize coefficients: β_orig_j = β_std_j / std_j
    beta_orig = beta_std / X_stds
    intercept = y_mean - float(X_means @ beta_orig)

    # Predictions and metrics
    y_pred = X_arr @ beta_orig + intercept
    residuals = y_arr - y_pred
    r2 = _r_squared(y_arr, y_pred)
    adj_r2 = _adj_r_squared(r2, n, p)
    mse = float(np.mean(residuals ** 2))

    coef_dict = {name: float(b) for name, b in zip(names, beta_orig)}
    std_coef = {name: float(b) for name, b in zip(names, beta_std)}
    selected = [name for name, b in zip(names, beta_orig) if abs(b) > 1e-10]

    return RegularizedResult(
        method="ridge",
        coefficients=coef_dict,
        intercept=intercept,
        alpha=float(alpha),
        l1_ratio=None,
        r_squared=r2,
        r_squared_adj=adj_r2,
        mse=mse,
        n_features_selected=len(selected),
        selected_features=selected,
        cv_score=cv_score,
        standardized_coef=std_coef,
        residuals=residuals.tolist(),
    )


def _ridge_cv_score(X: np.ndarray, y: np.ndarray, alpha: float, k: int = 5) -> float:
    """K-fold CV MSE for Ridge."""
    n = X.shape[0]
    p = X.shape[1]
    mse_sum = 0.0
    n_test = 0
    for train_idx, test_idx in _kfold_indices(n, k):
        X_tr, y_tr = X[train_idx], y[train_idx]
        X_te, y_te = X[test_idx], y[test_idx]
        XtX = X_tr.T @ X_tr
        beta = np.linalg.solve(XtX + alpha * np.eye(p), X_tr.T @ y_tr)
        pred = X_te @ beta
        mse_sum += np.sum((y_te - pred) ** 2)
        n_test += len(test_idx)
    return float(mse_sum / n_test)


def lasso(
    X: list[list[float]] | np.ndarray,
    y: list[float] | np.ndarray,
    alpha: float = 1.0,
    feature_names: list[str] | None = None,
    conf: float = 0.95,
    alphas: list[float] | np.ndarray | None = None,
    max_iter: int = 1000,
) -> RegularizedResult:
    """Lasso regression (L1 penalty) via coordinate descent.

    Args:
        X: Predictor matrix (n x p).
        y: Response vector (n,).
        alpha: Regularization strength. Ignored if alphas provided.
        feature_names: Names for predictors.
        conf: Confidence level (reserved).
        alphas: If provided, 5-fold CV to pick best alpha.
        max_iter: Maximum coordinate descent iterations.
    """
    X_arr = np.asarray(X, dtype=float)
    y_arr = np.asarray(y, dtype=float).ravel()
    n = len(y_arr)

    if X_arr.ndim == 1:
        X_arr = X_arr.reshape(-1, 1)

    p = X_arr.shape[1]
    names = feature_names or [f"X{i+1}" for i in range(p)]

    # Standardize X, center y
    X_std, X_means, X_stds = _standardize(X_arr)
    y_mean = float(np.mean(y_arr))
    y_c = y_arr - y_mean

    # Select alpha via CV if requested
    cv_score = None
    if alphas is not None:
        alpha, cv_score = _cv_select(X_std, y_c, alphas, l1_ratio=1.0, max_iter=max_iter)

    # Fit on standardized data
    beta_std = _coordinate_descent(X_std, y_c, alpha, l1_ratio=1.0, max_iter=max_iter)

    # Unstandardize
    beta_orig = beta_std / X_stds
    intercept = y_mean - float(X_means @ beta_orig)

    y_pred = X_arr @ beta_orig + intercept
    residuals = y_arr - y_pred
    r2 = _r_squared(y_arr, y_pred)
    adj_r2 = _adj_r_squared(r2, n, p)
    mse = float(np.mean(residuals ** 2))

    coef_dict = {name: float(b) for name, b in zip(names, beta_orig)}
    std_coef = {name: float(b) for name, b in zip(names, beta_std)}
    selected = [name for name, b in zip(names, beta_orig) if abs(b) > 1e-10]

    return RegularizedResult(
        method="lasso",
        coefficients=coef_dict,
        intercept=intercept,
        alpha=float(alpha),
        l1_ratio=1.0,
        r_squared=r2,
        r_squared_adj=adj_r2,
        mse=mse,
        n_features_selected=len(selected),
        selected_features=selected,
        cv_score=cv_score,
        standardized_coef=std_coef,
        residuals=residuals.tolist(),
    )


def elastic_net(
    X: list[list[float]] | np.ndarray,
    y: list[float] | np.ndarray,
    alpha: float = 1.0,
    l1_ratio: float = 0.5,
    feature_names: list[str] | None = None,
    conf: float = 0.95,
    max_iter: int = 1000,
) -> RegularizedResult:
    """Elastic Net regression (L1 + L2 penalty).

    Args:
        X: Predictor matrix (n x p).
        y: Response vector (n,).
        alpha: Overall regularization strength.
        l1_ratio: Mix between L1 and L2 (0=Ridge, 1=Lasso).
        feature_names: Names for predictors.
        conf: Confidence level (reserved).
        max_iter: Maximum coordinate descent iterations.
    """
    X_arr = np.asarray(X, dtype=float)
    y_arr = np.asarray(y, dtype=float).ravel()
    n = len(y_arr)

    if X_arr.ndim == 1:
        X_arr = X_arr.reshape(-1, 1)

    p = X_arr.shape[1]
    names = feature_names or [f"X{i+1}" for i in range(p)]

    X_std, X_means, X_stds = _standardize(X_arr)
    y_mean = float(np.mean(y_arr))
    y_c = y_arr - y_mean

    beta_std = _coordinate_descent(X_std, y_c, alpha, l1_ratio=l1_ratio, max_iter=max_iter)

    beta_orig = beta_std / X_stds
    intercept = y_mean - float(X_means @ beta_orig)

    y_pred = X_arr @ beta_orig + intercept
    residuals = y_arr - y_pred
    r2 = _r_squared(y_arr, y_pred)
    adj_r2 = _adj_r_squared(r2, n, p)
    mse = float(np.mean(residuals ** 2))

    coef_dict = {name: float(b) for name, b in zip(names, beta_orig)}
    std_coef = {name: float(b) for name, b in zip(names, beta_std)}
    selected = [name for name, b in zip(names, beta_orig) if abs(b) > 1e-10]

    return RegularizedResult(
        method="elastic_net",
        coefficients=coef_dict,
        intercept=intercept,
        alpha=float(alpha),
        l1_ratio=l1_ratio,
        r_squared=r2,
        r_squared_adj=adj_r2,
        mse=mse,
        n_features_selected=len(selected),
        selected_features=selected,
        cv_score=None,
        standardized_coef=std_coef,
        residuals=residuals.tolist(),
    )


def _cv_select(
    X: np.ndarray,
    y: np.ndarray,
    alphas: list[float] | np.ndarray,
    l1_ratio: float = 1.0,
    max_iter: int = 1000,
    k: int = 5,
) -> tuple[float, float]:
    """5-fold CV to pick best alpha for Lasso/Elastic Net. Returns (best_alpha, best_mse)."""
    n = X.shape[0]
    best_alpha = float(alphas[0]) if len(alphas) > 0 else 1.0
    best_mse = np.inf

    for a in alphas:
        mse_sum = 0.0
        n_test = 0
        for train_idx, test_idx in _kfold_indices(n, k):
            X_tr, y_tr = X[train_idx], y[train_idx]
            X_te, y_te = X[test_idx], y[test_idx]
            beta = _coordinate_descent(X_tr, y_tr, float(a), l1_ratio=l1_ratio, max_iter=max_iter)
            pred = X_te @ beta
            mse_sum += np.sum((y_te - pred) ** 2)
            n_test += len(test_idx)
        mse = mse_sum / n_test
        if mse < best_mse:
            best_mse = mse
            best_alpha = float(a)

    return best_alpha, float(best_mse)


def regularization_path(
    X: list[list[float]] | np.ndarray,
    y: list[float] | np.ndarray,
    method: str = "lasso",
    n_alphas: int = 50,
    feature_names: list[str] | None = None,
) -> PathResult:
    """Compute coefficient paths across a range of alpha values.

    Args:
        X: Predictor matrix (n x p).
        y: Response vector (n,).
        method: One of "lasso", "ridge", "elastic_net".
        n_alphas: Number of alpha values to evaluate.
        feature_names: Names for predictors.
    """
    X_arr = np.asarray(X, dtype=float)
    y_arr = np.asarray(y, dtype=float).ravel()

    if X_arr.ndim == 1:
        X_arr = X_arr.reshape(-1, 1)

    p = X_arr.shape[1]
    names = feature_names or [f"X{i+1}" for i in range(p)]

    X_std, X_means, X_stds = _standardize(X_arr)
    y_mean = float(np.mean(y_arr))
    y_c = y_arr - y_mean

    # Compute alpha_max: smallest alpha that zeros out all Lasso coefficients
    alpha_max = float(np.max(np.abs(X_std.T @ y_c))) / len(y_arr)
    if alpha_max == 0:
        alpha_max = 1.0
    alpha_min = alpha_max * 1e-4
    alphas = np.logspace(np.log10(alpha_max), np.log10(alpha_min), n_alphas)

    coef_paths: dict[str, list[float]] = {name: [] for name in names}

    # Track best alpha via simple CV on centered data
    best_alpha = float(alphas[0])
    best_mse = np.inf

    for a in alphas:
        if method == "ridge":
            XtX = X_std.T @ X_std
            beta_std = np.linalg.solve(XtX + a * np.eye(p), X_std.T @ y_c)
        elif method == "lasso":
            beta_std = _coordinate_descent(X_std, y_c, a, l1_ratio=1.0)
        elif method == "elastic_net":
            beta_std = _coordinate_descent(X_std, y_c, a, l1_ratio=0.5)
        else:
            raise ValueError(f"Unknown method: {method}. Use 'ridge', 'lasso', or 'elastic_net'.")

        for i, name in enumerate(names):
            coef_paths[name].append(float(beta_std[i]))

        # Simple MSE on training data for best-alpha selection
        pred = X_std @ beta_std
        mse = float(np.mean((y_c - pred) ** 2))
        # Prefer sparser solutions at similar MSE
        if mse < best_mse * 0.999:
            best_mse = mse
            best_alpha = float(a)

    return PathResult(
        alphas=alphas.tolist(),
        coefficients=coef_paths,
        best_alpha=best_alpha,
        method=method,
    )
