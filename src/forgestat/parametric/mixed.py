"""Mixed / Multilevel Models — Linear mixed-effects models.

Pure numpy/scipy. Critical for manufacturing where data is nested
(measurements within parts within machines within shifts).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy import optimize


@dataclass
class MixedResult:
    """Linear mixed-effects model result."""

    # Fixed effects
    fixed_effects: list[dict] = field(default_factory=list)

    # Random effects
    random_effects: list[dict] = field(default_factory=list)
    residual_variance: float = 0.0

    # Model fit
    log_likelihood: float = 0.0
    aic: float = 0.0
    bic: float = 0.0
    r_squared_marginal: float = 0.0
    r_squared_conditional: float = 0.0

    # ICC
    icc: float = 0.0

    # Diagnostics
    n_obs: int = 0
    n_groups: dict[str, int] = field(default_factory=dict)
    converged: bool = True

    # Interpretation helpers
    summary: str = ""


@dataclass
class VarianceDecomposition:
    """Variance decomposition across grouping factors."""

    components: list[dict] = field(default_factory=list)
    total: float = 0.0
    icc_values: dict[str, float] = field(default_factory=dict)


def _extract_column(data: list[dict], col: str) -> np.ndarray:
    """Extract a column from list-of-dicts data."""
    return np.array([row[col] for row in data])


def _dummy_encode(values: np.ndarray) -> tuple[np.ndarray, list]:
    """Create dummy (indicator) matrix for categorical values.

    Returns (Z matrix, unique_levels).
    """
    levels = sorted(set(values))
    n = len(values)
    Z = np.zeros((n, len(levels)))
    level_idx = {lev: i for i, lev in enumerate(levels)}
    for i, v in enumerate(values):
        Z[i, level_idx[v]] = 1.0
    return Z, levels


def _fixed_effects_matrix(data: list[dict], fixed: list[str]) -> tuple[np.ndarray, list[str]]:
    """Build fixed-effects design matrix with intercept.

    For categorical variables: dummy encoding (dropping first level).
    For numeric variables: use directly.
    """
    n = len(data)
    X_parts = [np.ones((n, 1))]  # intercept
    col_names = ["Intercept"]

    for f in fixed:
        vals = _extract_column(data, f)
        # Check if numeric
        try:
            numeric_vals = vals.astype(float)
            X_parts.append(numeric_vals.reshape(-1, 1))
            col_names.append(f)
        except (ValueError, TypeError):
            # Categorical: dummy encode, drop first
            levels = sorted(set(vals))
            for lev in levels[1:]:
                col = (vals == lev).astype(float)
                X_parts.append(col.reshape(-1, 1))
                col_names.append(f"{f}:{lev}")

    return np.hstack(X_parts), col_names


def mixed_model(
    data: list[dict],
    response: str,
    fixed: list[str],
    random: list[str],
    alpha: float = 0.05,
) -> MixedResult:
    """Fit a linear mixed-effects model.

    Uses REML estimation with Method of Moments for variance components
    and GLS for fixed effects.

    Args:
        data: List of row dicts (Django queryset style).
        response: Response variable column name.
        fixed: Fixed effect factor names.
        random: Random effect factor names (grouping variables).
        alpha: Significance level.

    Returns:
        MixedResult with fixed/random effects, fit statistics, ICC.
    """
    n = len(data)
    y = _extract_column(data, response).astype(float)
    y_mean = np.mean(y)

    # Build fixed-effects design matrix
    X, fixed_names = _fixed_effects_matrix(data, fixed)

    # Build random-effects design matrices
    Z_list = []
    random_levels = {}
    for rf in random:
        vals = _extract_column(data, rf)
        Z, levels = _dummy_encode(vals)
        Z_list.append((rf, Z, levels))
        random_levels[rf] = levels

    # Estimate variance components using Method of Moments (ANOVA-type)
    # For each random factor, compute between-group and within-group MS
    var_components = {}
    for rf, Z, levels in Z_list:
        k = len(levels)
        group_means = []
        group_ns = []
        group_vals = _extract_column(data, rf)
        for lev in levels:
            mask = group_vals == lev
            group_means.append(np.mean(y[mask]))
            group_ns.append(int(np.sum(mask)))

        group_means = np.array(group_means)
        group_ns = np.array(group_ns)
        grand_mean = np.mean(y)

        # SS between
        ss_between = float(np.sum(group_ns * (group_means - grand_mean) ** 2))
        # SS within
        ss_within = 0.0
        for lev, gm in zip(levels, group_means):
            mask = group_vals == lev
            ss_within += float(np.sum((y[mask] - gm) ** 2))

        df_between = k - 1
        df_within = n - k
        ms_between = ss_between / df_between if df_between > 0 else 0.0
        ms_within = ss_within / df_within if df_within > 0 else 0.0

        # n0: harmonic-ish average group size for unbalanced designs
        if k > 1:
            n0 = (n - np.sum(group_ns ** 2) / n) / (k - 1)
        else:
            n0 = float(np.mean(group_ns))

        # Between-group variance estimate
        sigma2_b = max(0.0, (ms_between - ms_within) / n0) if n0 > 0 else 0.0
        var_components[rf] = sigma2_b

    # Residual variance = MS_within from the finest grouping
    # Use the last random factor for residual estimation
    if Z_list:
        _, _, last_levels = Z_list[-1]
        last_vals = _extract_column(data, random[-1])
        ss_w = 0.0
        df_w = 0
        for lev in last_levels:
            mask = last_vals == lev
            g_data = y[mask]
            if len(g_data) > 1:
                ss_w += float(np.sum((g_data - np.mean(g_data)) ** 2))
                df_w += len(g_data) - 1
        sigma2_e = ss_w / df_w if df_w > 0 else float(np.var(y, ddof=1))
    else:
        sigma2_e = float(np.var(y, ddof=1))

    # Build V = sum(sigma2_b * Z Z') + sigma2_e * I
    V = sigma2_e * np.eye(n)
    for rf, Z, levels in Z_list:
        V += var_components[rf] * (Z @ Z.T)

    # GLS fixed effects: beta = (X' V^-1 X)^-1 X' V^-1 y
    try:
        V_inv = np.linalg.inv(V)
    except np.linalg.LinAlgError:
        V_inv = np.linalg.pinv(V)

    XtVinvX = X.T @ V_inv @ X
    try:
        XtVinvX_inv = np.linalg.inv(XtVinvX)
    except np.linalg.LinAlgError:
        XtVinvX_inv = np.linalg.pinv(XtVinvX)

    beta = XtVinvX_inv @ X.T @ V_inv @ y

    # Standard errors of fixed effects
    se_beta = np.sqrt(np.maximum(np.diag(XtVinvX_inv), 0))

    # t-statistics and p-values (Satterthwaite df approximation: use n - rank(X))
    df_fixed = max(n - X.shape[1], 1)
    from scipy import stats as sp_stats

    fixed_effects = []
    for i, name in enumerate(fixed_names):
        b = float(beta[i])
        se = float(se_beta[i])
        t_val = b / se if se > 0 else 0.0
        p_val = float(2 * (1 - sp_stats.t.cdf(abs(t_val), df_fixed)))
        fixed_effects.append({
            "name": name,
            "estimate": b,
            "se": se,
            "t": t_val,
            "p": p_val,
            "significant": p_val < alpha,
        })

    # Random effects output
    total_var = sum(var_components.values()) + sigma2_e
    random_effects = []
    for rf in random:
        v = var_components.get(rf, 0.0)
        icc_val = v / total_var if total_var > 0 else 0.0
        random_effects.append({
            "name": rf,
            "variance": v,
            "std": float(np.sqrt(v)),
            "icc": icc_val,
        })

    # Overall ICC (proportion of variance at group level)
    total_between = sum(var_components.values())
    icc = total_between / total_var if total_var > 0 else 0.0

    # Log-likelihood (REML)
    residuals = y - X @ beta
    sign, log_det_V = np.linalg.slogdet(V)
    if sign <= 0:
        log_det_V = 0.0
    ll = -0.5 * (n * np.log(2 * np.pi) + log_det_V + float(residuals.T @ V_inv @ residuals))

    # AIC, BIC
    n_params = X.shape[1] + len(random) + 1  # fixed + variance components + residual
    aic = -2 * ll + 2 * n_params
    bic = -2 * ll + n_params * np.log(n)

    # R-squared: marginal and conditional (Nakagawa & Schielzeth)
    # Marginal: variance explained by fixed effects / total
    y_fixed = X @ beta
    var_fixed = float(np.var(y_fixed, ddof=0))
    r2_marginal = var_fixed / (var_fixed + total_between + sigma2_e) if (var_fixed + total_var) > 0 else 0.0
    # Conditional: (fixed + random) / total
    r2_conditional = (var_fixed + total_between) / (var_fixed + total_between + sigma2_e) if (var_fixed + total_var) > 0 else 0.0

    # N groups
    n_groups = {rf: len(levels) for rf, (_, levels) in zip(random, [(Z, l) for _, Z, l in Z_list])}

    # Summary text
    summary_lines = [
        f"Linear Mixed Model: {response} ~ {' + '.join(fixed)} | {' + '.join(random)}",
        f"Observations: {n}",
    ]
    for rf in random:
        summary_lines.append(f"Groups ({rf}): {n_groups[rf]}")
    summary_lines.append(f"ICC: {icc:.4f}")
    summary_lines.append(f"Marginal R²: {r2_marginal:.4f}, Conditional R²: {r2_conditional:.4f}")
    summary_lines.append("")
    summary_lines.append("Fixed Effects:")
    for fe in fixed_effects:
        sig = "*" if fe["significant"] else ""
        summary_lines.append(f"  {fe['name']}: {fe['estimate']:.4f} (SE={fe['se']:.4f}, t={fe['t']:.2f}, p={fe['p']:.4f}){sig}")
    summary_lines.append("")
    summary_lines.append("Random Effects:")
    for re in random_effects:
        summary_lines.append(f"  {re['name']}: var={re['variance']:.4f}, std={re['std']:.4f}, ICC={re['icc']:.4f}")
    summary_lines.append(f"  Residual: var={sigma2_e:.4f}, std={np.sqrt(sigma2_e):.4f}")

    return MixedResult(
        fixed_effects=fixed_effects,
        random_effects=random_effects,
        residual_variance=sigma2_e,
        log_likelihood=float(ll),
        aic=float(aic),
        bic=float(bic),
        r_squared_marginal=r2_marginal,
        r_squared_conditional=r2_conditional,
        icc=icc,
        n_obs=n,
        n_groups=n_groups,
        converged=True,
        summary="\n".join(summary_lines),
    )


def variance_decomposition(
    data: list[dict],
    response: str,
    factors: list[str],
) -> VarianceDecomposition:
    """Decompose total variance into between-group and within-group components.

    Uses hierarchical (sequential) ANOVA-type estimation: each factor is
    evaluated on the residuals after removing the effects of previous factors,
    so nested/overlapping factors do not double-count variance.

    Args:
        data: List of row dicts.
        response: Response variable column name.
        factors: Grouping factor names (from outermost to innermost).
    """
    n = len(data)
    y = _extract_column(data, response).astype(float)
    total_var = float(np.var(y, ddof=1))

    components = []
    # Work on residuals sequentially — each factor explains variance
    # remaining after previous factors have been accounted for
    residuals = y - np.mean(y)  # center

    for factor in factors:
        vals = _extract_column(data, factor)
        levels = sorted(set(vals))
        k = len(levels)

        # Compute group means of current residuals
        group_means = {}
        group_ns = {}
        for lev in levels:
            mask = vals == lev
            group_means[lev] = float(np.mean(residuals[mask]))
            group_ns[lev] = int(np.sum(mask))

        # SS explained by this factor on current residuals
        ss_between = sum(
            group_ns[lev] * group_means[lev] ** 2 for lev in levels
        )
        ss_resid_before = float(np.sum(residuals ** 2))

        # Variance component estimate
        df_between = k - 1
        n_arr = np.array([group_ns[lev] for lev in levels])
        if k > 1:
            n0 = (n - np.sum(n_arr ** 2) / n) / (k - 1)
        else:
            n0 = float(np.mean(n_arr))

        # Remove this factor's effect from residuals
        for lev in levels:
            mask = vals == lev
            residuals[mask] -= group_means[lev]

        ss_resid_after = float(np.sum(residuals ** 2))
        df_within = n - k  # approximate
        ms_within = ss_resid_after / max(df_within, 1)
        ms_between = ss_between / max(df_between, 1)

        sigma2_b = max(0.0, (ms_between - ms_within) / n0) if n0 > 0 else 0.0

        pct = (sigma2_b / total_var * 100) if total_var > 0 else 0.0
        components.append({
            "source": factor,
            "variance": sigma2_b,
            "percent": pct,
            "label": f"Between-{factor}",
        })

    # Residual variance: within the finest grouping factor
    last_factor = factors[-1]
    last_vals = _extract_column(data, last_factor)
    last_levels = sorted(set(last_vals))
    ss_w = 0.0
    df_w = 0
    for lev in last_levels:
        mask = last_vals == lev
        g_data = y[mask]
        if len(g_data) > 1:
            ss_w += float(np.sum((g_data - np.mean(g_data)) ** 2))
            df_w += len(g_data) - 1
    residual_var = ss_w / df_w if df_w > 0 else float(np.var(y, ddof=1))

    components.append({
        "source": "Residual",
        "variance": residual_var,
        "percent": 0.0,  # filled below
        "label": "Within-group",
    })

    # Recompute total from components to ensure consistency
    comp_total = sum(c["variance"] for c in components)

    # Recompute percentages relative to component total (sums to 100%)
    for comp in components:
        comp["percent"] = (comp["variance"] / comp_total * 100) if comp_total > 0 else 0.0

    # ICC values: each factor's proportion of total
    icc_values = {}
    for comp in components[:-1]:  # exclude residual
        icc_values[comp["source"]] = comp["variance"] / comp_total if comp_total > 0 else 0.0

    return VarianceDecomposition(
        components=components,
        total=total_var,
        icc_values=icc_values,
    )


def nested_anova(
    data: list[dict],
    response: str,
    factors: list[str],
    alpha: float = 0.05,
) -> MixedResult:
    """Nested ANOVA — convenience wrapper for common manufacturing pattern.

    e.g., Parts nested within Machines:
        nested_anova(data, "diameter", ["machine", "part"])

    The first factor is treated as the main random grouping factor.
    Subsequent factors are treated as nested within the first.

    Args:
        data: List of row dicts.
        response: Response variable column name.
        factors: Factor names from outermost to innermost.
        alpha: Significance level.
    """
    if len(factors) < 1:
        raise ValueError("Need at least one factor for nested ANOVA")

    # Treat all factors as random effects, no fixed effects
    # Use mixed_model with empty fixed effects and all random
    return mixed_model(
        data=data,
        response=response,
        fixed=[],
        random=factors,
        alpha=alpha,
    )
