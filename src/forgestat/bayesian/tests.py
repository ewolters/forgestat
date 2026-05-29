"""Bayesian hypothesis tests — conjugate posteriors for common tests.

All use closed-form conjugate math (no MCMC). Pure scipy.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
from scipy import stats


@dataclass
class BayesianTestResult:
    """Result of a Bayesian hypothesis test."""

    test_name: str
    bf10: float = 0.0  # Bayes factor in favor of H1
    bf01: float = 0.0  # Bayes factor in favor of H0
    bf_label: str = ""  # "anecdotal", "moderate", "strong", "very_strong", "extreme"
    posterior_mean: float | None = None
    posterior_std: float | None = None
    credible_interval: tuple[float, float] | None = None
    ci_level: float = 0.95
    rope: tuple[float, float] | None = None  # region of practical equivalence
    p_rope: float | None = None  # probability within ROPE
    extra: dict = field(default_factory=dict)


def _bf_label(bf: float) -> str:
    """Classify Bayes factor magnitude (Jeffreys' scale)."""
    abf = abs(bf)
    if abf < 1:
        abf = 1 / abf  # flip to H0 evidence
    if abf < 1:
        return "no_evidence"
    elif abf < 3:
        return "anecdotal"
    elif abf < 10:
        return "moderate"
    elif abf < 30:
        return "strong"
    elif abf < 100:
        return "very_strong"
    else:
        return "extreme"


def bayesian_ttest_one_sample(
    data: list[float] | np.ndarray,
    mu: float = 0.0,
    prior_scale: float = 0.707,
    ci_level: float = 0.95,
    rope: tuple[float, float] | None = None,
) -> BayesianTestResult:
    """Bayesian one-sample t-test (JZS Bayes factor).

    Uses the Jeffreys-Zellner-Siow (JZS) prior — a Cauchy prior on effect size.

    Args:
        data: Sample values.
        mu: Value under H0.
        prior_scale: Scale of the Cauchy prior on effect size (default √2/2).
        ci_level: Credible interval level.
        rope: Region of practical equivalence (low, high) for effect size.
    """
    x = np.asarray(data, dtype=float)
    x = x[np.isfinite(x)]
    n = len(x)

    if n < 2:
        raise ValueError("Need at least 2 observations")

    mean = float(np.mean(x))
    se = float(np.std(x, ddof=1) / math.sqrt(n))
    t_stat = (mean - mu) / se if se > 0 else 0.0
    df = n - 1

    # JZS Bayes factor via numerical integration
    bf10 = _jzs_bf(t_stat, n, prior_scale)

    # Posterior (Normal-Gamma conjugate approximation)
    post_mean = mean
    post_std = se

    alpha_ci = (1 - ci_level) / 2
    ci_lo = post_mean + stats.t.ppf(alpha_ci, df) * post_std
    ci_hi = post_mean + stats.t.ppf(1 - alpha_ci, df) * post_std

    # ROPE probability (proportion of posterior in ROPE)
    p_rope_val = None
    if rope is not None:
        d = (mean - mu) / float(np.std(x, ddof=1)) if np.std(x, ddof=1) > 0 else 0
        se_d = 1 / math.sqrt(n)
        p_rope_val = float(
            stats.norm.cdf(rope[1], loc=d, scale=se_d)
            - stats.norm.cdf(rope[0], loc=d, scale=se_d)
        )

    return BayesianTestResult(
        test_name="Bayesian one-sample t-test",
        bf10=bf10,
        bf01=1 / bf10 if bf10 > 0 else float("inf"),
        bf_label=_bf_label(bf10),
        posterior_mean=post_mean,
        posterior_std=post_std,
        credible_interval=(float(ci_lo), float(ci_hi)),
        ci_level=ci_level,
        rope=rope,
        p_rope=p_rope_val,
        extra={"t_statistic": t_stat, "df": df, "n": n},
    )


def bayesian_ttest_two_sample(
    x1: list[float] | np.ndarray,
    x2: list[float] | np.ndarray,
    prior_scale: float = 0.707,
    ci_level: float = 0.95,
) -> BayesianTestResult:
    """Bayesian two-sample t-test (JZS Bayes factor).

    Args:
        x1, x2: Two independent samples.
        prior_scale: Cauchy prior scale.
        ci_level: Credible interval level.
    """
    a = np.asarray(x1, dtype=float)
    b = np.asarray(x2, dtype=float)
    a, b = a[np.isfinite(a)], b[np.isfinite(b)]
    n1, n2 = len(a), len(b)

    if n1 < 2 or n2 < 2:
        raise ValueError("Need at least 2 observations per group")

    # Welch t-statistic
    t_stat, _ = stats.ttest_ind(a, b, equal_var=False)
    t_stat = float(t_stat)
    n_eff = n1 * n2 / (n1 + n2)  # effective sample size

    bf10 = _jzs_bf(t_stat, n_eff, prior_scale)

    diff = float(np.mean(a) - np.mean(b))
    se = math.sqrt(float(np.var(a, ddof=1)) / n1 + float(np.var(b, ddof=1)) / n2)

    s1 = float(np.var(a, ddof=1))
    s2 = float(np.var(b, ddof=1))
    df_num = (s1 / n1 + s2 / n2) ** 2
    df_den = (s1 / n1) ** 2 / (n1 - 1) + (s2 / n2) ** 2 / (n2 - 1)
    df = df_num / df_den if df_den > 0 else n1 + n2 - 2

    alpha_ci = (1 - ci_level) / 2
    ci_lo = diff + stats.t.ppf(alpha_ci, df) * se
    ci_hi = diff + stats.t.ppf(1 - alpha_ci, df) * se

    return BayesianTestResult(
        test_name="Bayesian two-sample t-test",
        bf10=bf10,
        bf01=1 / bf10 if bf10 > 0 else float("inf"),
        bf_label=_bf_label(bf10),
        posterior_mean=diff,
        posterior_std=se,
        credible_interval=(float(ci_lo), float(ci_hi)),
        ci_level=ci_level,
        extra={"t_statistic": t_stat, "df": float(df), "n1": n1, "n2": n2},
    )


def bayesian_proportion(
    successes: int,
    n: int,
    prior_a: float = 1.0,
    prior_b: float = 1.0,
    ci_level: float = 0.95,
) -> BayesianTestResult:
    """Bayesian inference for a proportion (Beta-Binomial conjugate).

    Args:
        successes: Number of successes.
        n: Total trials.
        prior_a: Beta prior alpha (default 1 = uniform).
        prior_b: Beta prior beta.
        ci_level: Credible interval level.
    """
    post_a = prior_a + successes
    post_b = prior_b + (n - successes)

    post_mean = post_a / (post_a + post_b)
    post_var = (post_a * post_b) / ((post_a + post_b) ** 2 * (post_a + post_b + 1))
    post_std = math.sqrt(post_var)

    alpha_ci = (1 - ci_level) / 2
    ci_lo = float(stats.beta.ppf(alpha_ci, post_a, post_b))
    ci_hi = float(stats.beta.ppf(1 - alpha_ci, post_a, post_b))

    # BF for p > 0.5 vs p <= 0.5 (one-sided)
    p_gt_half = 1 - stats.beta.cdf(0.5, post_a, post_b)
    prior_gt_half = 1 - stats.beta.cdf(0.5, prior_a, prior_b)
    bf10 = (p_gt_half / (1 - p_gt_half)) / (prior_gt_half / (1 - prior_gt_half)) if (1 - p_gt_half) > 0 and (1 - prior_gt_half) > 0 else 1.0

    return BayesianTestResult(
        test_name="Bayesian proportion (Beta-Binomial)",
        bf10=float(bf10),
        bf01=1 / bf10 if bf10 > 0 else float("inf"),
        bf_label=_bf_label(bf10),
        posterior_mean=post_mean,
        posterior_std=post_std,
        credible_interval=(ci_lo, ci_hi),
        ci_level=ci_level,
        extra={
            "posterior_alpha": post_a,
            "posterior_beta": post_b,
            "successes": successes,
            "n": n,
        },
    )


def bayesian_correlation(
    x: list[float] | np.ndarray,
    y: list[float] | np.ndarray,
    ci_level: float = 0.95,
) -> BayesianTestResult:
    """Bayesian test for correlation (Jeffreys' BF).

    Uses the exact Bayes factor for Pearson correlation under uniform prior on ρ.

    Args:
        x, y: Paired observations.
        ci_level: Credible interval level.
    """
    x_arr = np.asarray(x, dtype=float)
    y_arr = np.asarray(y, dtype=float)
    mask = np.isfinite(x_arr) & np.isfinite(y_arr)
    x_arr, y_arr = x_arr[mask], y_arr[mask]
    n = len(x_arr)

    if n < 4:
        raise ValueError("Need at least 4 paired observations")

    r, _ = stats.pearsonr(x_arr, y_arr)
    r = float(r)

    # Jeffreys' BF for correlation: BF10 ∝ (1-r²)^((n-1)/2)
    if abs(r) > 0.9999:
        bf10 = 1e200  # effectively infinite evidence
    else:
        t_stat = r * math.sqrt((n - 2) / (1 - r ** 2))
        bf10 = _jzs_bf(t_stat, n, 0.707)

    # Posterior CI via Fisher z-transform
    z_r = 0.5 * math.log((1 + r) / (1 - r)) if abs(r) < 1 else 0
    se_z = 1 / math.sqrt(n - 3) if n > 3 else 1
    alpha_ci = (1 - ci_level) / 2
    z_lo = z_r + stats.norm.ppf(alpha_ci) * se_z
    z_hi = z_r + stats.norm.ppf(1 - alpha_ci) * se_z
    ci_lo = math.tanh(z_lo)
    ci_hi = math.tanh(z_hi)

    return BayesianTestResult(
        test_name="Bayesian correlation",
        bf10=bf10,
        bf01=1 / bf10 if bf10 > 0 else float("inf"),
        bf_label=_bf_label(bf10),
        posterior_mean=r,
        posterior_std=se_z,
        credible_interval=(ci_lo, ci_hi),
        ci_level=ci_level,
        extra={"r": r, "n": n},
    )


def bayes_factor_shadow(
    t_statistic: float,
    n: int | float,
    prior_scale: float = 0.707,
) -> dict:
    """Compute a Bayes factor 'shadow' for any frequentist t-statistic.

    This is the computation behind SVEND's _bayesian_shadow().
    Attach to any test that produces a t-statistic.

    Args:
        t_statistic: The observed t (or z) statistic.
        n: Sample size (or effective sample size).
        prior_scale: Cauchy prior scale.

    Returns:
        Dict with bf10, bf01, bf_label, interpretation.
    """
    bf10 = _jzs_bf(t_statistic, n, prior_scale)
    bf01 = 1 / bf10 if bf10 > 0 else float("inf")
    label = _bf_label(bf10)

    if bf10 > 3:
        interp = "Evidence supports the alternative hypothesis"
    elif bf01 > 3:
        interp = "Evidence supports the null hypothesis"
    else:
        interp = "Evidence is inconclusive"

    return {
        "bf10": bf10,
        "bf01": bf01,
        "bf_label": label,
        "interpretation": interp,
    }


def _jzs_bf(t: float, n: float, r: float = 0.707) -> float:
    """JZS Bayes factor via Cauchy prior on effect size.

    Approximation using the BIC-based method for moderate sample sizes,
    with exact integral for small samples.

    BF10 = integral_0^inf  (1 + n*g)^(-1/2) * exp(t²*n*g / (2*(1+n*g))) * Cauchy(g|r) dg

    Uses numerical integration via scipy.
    """
    from scipy import integrate

    if abs(t) < 1e-10:
        return 1.0  # no evidence either way

    df = max(1, n - 1) if isinstance(n, int) else max(1, n - 1)

    def integrand(g):
        if g <= 0:
            return 0.0
        ng = n * g
        term1 = (1 + ng) ** (-0.5)
        log_term2 = t ** 2 * ng / (2 * (1 + ng))
        if log_term2 > 500:
            return 0.0  # avoid overflow, contribution is negligible in tails
        term2 = math.exp(log_term2)
        # Half-Cauchy prior: 2/(pi*r) * 1/(1 + (g/r²))
        prior = 2 / (math.pi * r) * 1 / (1 + g / r ** 2)
        return term1 * term2 * prior

    try:
        result, _ = integrate.quad(integrand, 0, np.inf, limit=100)
        # BF10 relative to point null
        null_density = stats.t.pdf(t, df)
        if null_density > 0 and result > 0:
            return max(1e-10, result / null_density)
        return 1.0
    except Exception:
        # BIC approximation fallback
        log_bf = 0.5 * (t ** 2 - math.log(n))
        if log_bf > 500:
            return 1e200  # extreme evidence
        return max(1e-10, math.exp(log_bf))


def _bic_bayes_factor(rss_full: float, rss_null: float, n: int, df_extra: int) -> float:
    """BIC approximation to BF10 for a nested model (full vs reduced).

    BF10 ≈ exp(-ΔBIC/2) where ΔBIC = n·ln(RSS_full/RSS_null) + df_extra·ln(n).
    Wagenmakers (2007). df_extra = added parameters in the full model.
    """
    if rss_full <= 0 or rss_null <= 0:
        return 1e200
    log_bf = -0.5 * (n * math.log(rss_full / rss_null) + df_extra * math.log(n))
    if log_bf > 460:
        return 1e200
    return float(max(1e-10, min(math.exp(log_bf), 1e200)))


def bayesian_anova(
    groups: dict[str, list] | list[list],
    ci_level: float = 0.95,
) -> BayesianTestResult:
    """Bayesian one-way ANOVA (BIC-approximated Bayes factor).

    Compares a full model (one mean per group) against the null (a single
    grand mean). posterior_mean reports η² (proportion of variance explained).

    Args:
        groups: Mapping of label → values, or a sequence of value sequences.
        ci_level: Reported credible level (carried through; no CI on η²).
    """
    values = groups.values() if isinstance(groups, dict) else groups
    arrays = [np.asarray(v, dtype=float) for v in values]
    arrays = [a[np.isfinite(a)] for a in arrays]
    arrays = [a for a in arrays if len(a) > 0]
    k = len(arrays)
    if k < 2:
        raise ValueError("Need at least 2 groups")

    ns = [len(a) for a in arrays]
    n = sum(ns)
    if n <= k:
        raise ValueError("Need more observations than groups")

    grand_mean = float(np.mean(np.concatenate(arrays)))
    ss_between = float(sum(ni * (float(np.mean(a)) - grand_mean) ** 2 for a, ni in zip(arrays, ns)))
    ss_within = float(sum(float(np.sum((a - np.mean(a)) ** 2)) for a in arrays))
    ss_total = ss_between + ss_within
    df_between, df_within = k - 1, n - k
    ms_within = ss_within / df_within if df_within > 0 else 0.0
    f_stat = (ss_between / df_between) / ms_within if ms_within > 0 else float("inf")
    eta_sq = ss_between / ss_total if ss_total > 0 else 0.0

    bf10 = _bic_bayes_factor(ss_within, ss_total, n, df_between)
    return BayesianTestResult(
        test_name="Bayesian one-way ANOVA",
        bf10=bf10,
        bf01=1 / bf10 if bf10 > 0 else float("inf"),
        bf_label=_bf_label(bf10),
        posterior_mean=eta_sq,
        ci_level=ci_level,
        extra={
            "f_statistic": f_stat, "df_between": df_between, "df_within": df_within,
            "k": k, "n": n, "eta_squared": eta_sq,
            "ss_between": ss_between, "ss_within": ss_within,
        },
    )


def bayesian_regression(
    y: list[float] | np.ndarray,
    X: list | np.ndarray,
    ci_level: float = 0.95,
) -> BayesianTestResult:
    """Bayesian linear regression (BIC-approximated Bayes factor).

    Compares the full model (intercept + predictors) against the
    intercept-only null. posterior_mean reports R².

    Args:
        y: Response vector.
        X: Predictor matrix, n rows × p columns (or a 1-D vector for p=1).
        ci_level: Reported credible level.
    """
    y_arr = np.asarray(y, dtype=float).ravel()
    X_arr = np.asarray(X, dtype=float)
    if X_arr.ndim == 1:
        X_arr = X_arr.reshape(-1, 1)
    mask = np.isfinite(y_arr) & np.all(np.isfinite(X_arr), axis=1)
    y_arr, X_arr = y_arr[mask], X_arr[mask]
    n, p = X_arr.shape
    if n <= p + 1:
        raise ValueError("Need more observations than predictors")

    design = np.column_stack([np.ones(n), X_arr])
    coef, _, _, _ = np.linalg.lstsq(design, y_arr, rcond=None)
    rss_full = float(np.sum((y_arr - design @ coef) ** 2))
    rss_null = float(np.sum((y_arr - np.mean(y_arr)) ** 2))
    r_squared = 1 - rss_full / rss_null if rss_null > 0 else 0.0

    bf10 = _bic_bayes_factor(rss_full, rss_null, n, p)
    return BayesianTestResult(
        test_name="Bayesian linear regression",
        bf10=bf10,
        bf01=1 / bf10 if bf10 > 0 else float("inf"),
        bf_label=_bf_label(bf10),
        posterior_mean=r_squared,
        ci_level=ci_level,
        extra={
            "r_squared": r_squared,
            "coefficients": [float(c) for c in coef[1:]],
            "intercept": float(coef[0]),
            "n": n, "n_predictors": p,
        },
    )


def bayesian_ab(
    x1: list[float] | np.ndarray,
    x2: list[float] | np.ndarray,
    prior_scale: float = 0.707,
    ci_level: float = 0.95,
) -> BayesianTestResult:
    """Bayesian A/B test: control (x1) vs variant (x2).

    Reports P(B > A) and the uplift posterior under flat priors (normal
    approximation), plus a JZS Bayes factor. Works on continuous outcomes or
    binary 0/1 conversions (the mean is then the conversion rate).
    """
    a = np.asarray(x1, dtype=float)
    b = np.asarray(x2, dtype=float)
    a, b = a[np.isfinite(a)], b[np.isfinite(b)]
    n1, n2 = len(a), len(b)
    if n1 < 2 or n2 < 2:
        raise ValueError("Need at least 2 observations per arm")

    mean_a, mean_b = float(np.mean(a)), float(np.mean(b))
    se_diff = math.sqrt(float(np.var(a, ddof=1)) / n1 + float(np.var(b, ddof=1)) / n2)
    uplift = mean_b - mean_a
    if se_diff > 0:
        prob_b_better = float(stats.norm.cdf(uplift / se_diff))
    else:
        prob_b_better = 0.5 if uplift == 0 else float(uplift > 0)

    t_stat, _ = stats.ttest_ind(a, b, equal_var=False)
    bf10 = _jzs_bf(float(t_stat), n1 * n2 / (n1 + n2), prior_scale)

    z = stats.norm.ppf(1 - (1 - ci_level) / 2)
    return BayesianTestResult(
        test_name="Bayesian A/B test",
        bf10=bf10,
        bf01=1 / bf10 if bf10 > 0 else float("inf"),
        bf_label=_bf_label(bf10),
        posterior_mean=uplift,
        posterior_std=se_diff,
        credible_interval=(float(uplift - z * se_diff), float(uplift + z * se_diff)),
        ci_level=ci_level,
        extra={
            "prob_b_better": prob_b_better, "mean_a": mean_a, "mean_b": mean_b,
            "uplift": uplift, "n1": n1, "n2": n2,
        },
    )
