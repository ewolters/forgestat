"""Multivariate analysis — PCA, Hotelling T², MANOVA.

PCA uses numpy eigendecomposition. No sklearn required.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
from scipy import stats


@dataclass
class PCAResult:
    """Principal Component Analysis result."""

    n_components: int = 0
    eigenvalues: list[float] = field(default_factory=list)
    variance_explained: list[float] = field(default_factory=list)  # proportion per PC
    cumulative_variance: list[float] = field(default_factory=list)
    loadings: dict[str, list[float]] = field(default_factory=dict)  # {variable: [load_pc1, load_pc2, ...]}
    scores: list[list[float]] = field(default_factory=list)  # n × n_components
    n: int = 0
    p: int = 0


@dataclass
class HotellingResult:
    """Hotelling's T² test result."""

    t2_statistic: float = 0.0
    f_statistic: float = 0.0
    p_value: float = 0.0
    df1: int = 0
    df2: int = 0


@dataclass
class ManovaResult:
    """One-way MANOVA result."""

    wilks_lambda: float = 0.0
    f_statistic: float = 0.0
    p_value: float = 0.0
    df1: float = 0.0
    df2: float = 0.0
    pillai_trace: float = 0.0


def pca(
    data: dict[str, list[float]],
    n_components: int | None = None,
    standardize: bool = True,
) -> PCAResult:
    """Principal Component Analysis via eigendecomposition.

    Args:
        data: Dict of variable_name → values (all same length).
        n_components: Number of components to retain (default: all).
        standardize: If True, use correlation matrix. If False, covariance.

    Returns:
        PCAResult with eigenvalues, loadings, scores, variance explained.
    """
    names = list(data.keys())
    X = np.column_stack([np.asarray(data[k], dtype=float) for k in names])
    n, p = X.shape

    # Center (and optionally scale)
    means = np.mean(X, axis=0)
    X_centered = X - means

    if standardize:
        stds = np.std(X, axis=0, ddof=1)
        stds[stds == 0] = 1
        X_centered = X_centered / stds
        cov_matrix = np.corrcoef(X.T)
    else:
        cov_matrix = np.cov(X.T)

    # Eigendecomposition
    eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)

    # Sort descending
    idx = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]

    # Retain components
    k = n_components or p
    k = min(k, p)

    total_var = float(np.sum(eigenvalues))
    var_explained = [float(e / total_var) for e in eigenvalues[:k]]
    cumulative = []
    cumsum = 0.0
    for v in var_explained:
        cumsum += v
        cumulative.append(cumsum)

    # Loadings (eigenvectors scaled by sqrt of eigenvalue)
    loadings = {}
    for i, name in enumerate(names):
        loadings[name] = [float(eigenvectors[i, j]) for j in range(k)]

    # Scores
    scores = (X_centered @ eigenvectors[:, :k]).tolist()

    return PCAResult(
        n_components=k,
        eigenvalues=[float(e) for e in eigenvalues[:k]],
        variance_explained=var_explained,
        cumulative_variance=cumulative,
        loadings=loadings,
        scores=scores,
        n=n,
        p=p,
    )


def hotelling_t2_one_sample(
    data: dict[str, list[float]],
    mu: list[float] | None = None,
) -> HotellingResult:
    """Hotelling's T² test: multivariate generalization of one-sample t-test.

    H₀: μ = μ₀ (default: zero vector).

    Args:
        data: Dict of variable_name → values.
        mu: Hypothesized mean vector (default: zeros).

    Returns:
        HotellingResult with T², F-statistic, p-value.
    """
    names = list(data.keys())
    X = np.column_stack([np.asarray(data[k], dtype=float) for k in names])
    n, p = X.shape

    if mu is None:
        mu0 = np.zeros(p)
    else:
        mu0 = np.asarray(mu, dtype=float)

    x_bar = np.mean(X, axis=0)
    S = np.cov(X.T)
    if S.ndim == 0:
        S = np.array([[S]])

    diff = x_bar - mu0

    try:
        S_inv = np.linalg.inv(S)
    except np.linalg.LinAlgError:
        S_inv = np.linalg.pinv(S)

    t2 = float(n * diff @ S_inv @ diff)

    # Convert to F
    f_stat = t2 * (n - p) / (p * (n - 1))
    df1 = p
    df2 = n - p

    p_value = float(1 - stats.f.cdf(f_stat, df1, df2)) if df2 > 0 else 1.0

    return HotellingResult(
        t2_statistic=t2,
        f_statistic=float(f_stat),
        p_value=p_value,
        df1=df1,
        df2=df2,
    )


def one_way_manova(
    data: dict[str, list[float]],
    groups: list[str | int],
) -> ManovaResult:
    """One-way MANOVA: multivariate generalization of one-way ANOVA.

    H₀: All group mean vectors are equal.

    Args:
        data: Dict of response_variable → values (all same length).
        groups: Group label for each observation.

    Returns:
        ManovaResult with Wilks' lambda, Pillai's trace, F, p-value.
    """
    names = list(data.keys())
    Y = np.column_stack([np.asarray(data[k], dtype=float) for k in names])
    g_arr = np.asarray(groups)
    n, p = Y.shape
    unique_groups = np.unique(g_arr)
    k = len(unique_groups)

    grand_mean = np.mean(Y, axis=0)

    # Between-group (H) and within-group (E) SS&CP matrices
    H = np.zeros((p, p))
    E = np.zeros((p, p))

    for grp in unique_groups:
        mask = g_arr == grp
        Y_g = Y[mask]
        n_g = len(Y_g)
        mean_g = np.mean(Y_g, axis=0)
        diff = (mean_g - grand_mean).reshape(-1, 1)
        H += n_g * (diff @ diff.T)
        E += (Y_g - mean_g).T @ (Y_g - mean_g)

    # Wilks' Lambda = det(E) / det(E + H)
    # Use eigenvalues for numerical stability
    try:
        E_inv_H = np.linalg.solve(E, H)
        eigvals = np.real(np.linalg.eigvals(E_inv_H))
        wilks = float(np.prod(1 / (1 + eigvals)))
    except np.linalg.LinAlgError:
        try:
            det_eh = np.linalg.det(E + H)
            wilks = np.linalg.det(E) / det_eh if det_eh != 0 else 0.0
        except Exception:
            wilks = 1.0

    if not np.isfinite(wilks):
        wilks = 0.0  # extreme separation
    wilks = max(0.0, min(1.0, float(wilks)))

    df_h = k - 1

    # Pillai's trace = trace(H(H+E)^-1)
    try:
        HE_inv = np.linalg.solve(E + H, H)
        pillai = float(np.trace(HE_inv))
        if not np.isfinite(pillai):
            pillai = float(min(p, df_h))  # max possible
    except np.linalg.LinAlgError:
        pillai = 0.0

    # Rao's F approximation for Wilks' Lambda
    df_e = n - k
    s = min(p, df_h)

    if wilks == 0.0:
        # Perfect separation — extreme significance
        f_stat = 1e6
        df1_f = float(p * df_h)
        df2_f = float(max(1, df_e))
        p_value = 0.0
    elif 0 < wilks < 1 and s > 0:
        # Rao's F
        ms = df_e + df_h - (p + df_h + 1) / 2
        if p ** 2 + df_h ** 2 > 5:
            t_val = math.sqrt((p ** 2 * df_h ** 2 - 4) / (p ** 2 + df_h ** 2 - 5))
        else:
            t_val = 1.0

        df1_f = p * df_h
        df2_f = ms * t_val - p * df_h / 2 + 1

        if t_val > 0 and df2_f > 0:
            lambda_root = wilks ** (1 / t_val) if t_val != 0 else wilks
            if lambda_root < 1:
                f_stat = ((1 - lambda_root) / lambda_root) * (df2_f / df1_f)
            else:
                f_stat = 0.0
            p_value = float(1 - stats.f.cdf(max(0, f_stat), max(1, df1_f), max(1, df2_f)))
        else:
            f_stat = 0.0
            p_value = 1.0
    else:
        f_stat = 0.0
        df1_f = p * df_h
        df2_f = 1.0
        p_value = 1.0

    return ManovaResult(
        wilks_lambda=wilks,
        f_statistic=float(f_stat),
        p_value=p_value,
        df1=float(df1_f),
        df2=float(df2_f),
        pillai_trace=pillai,
    )
