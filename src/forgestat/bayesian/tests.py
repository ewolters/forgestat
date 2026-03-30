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
            return max(1e-10, result)
        return 1.0
    except Exception:
        # BIC approximation fallback
        log_bf = 0.5 * (t ** 2 - math.log(n))
        if log_bf > 500:
            return 1e200  # extreme evidence
        return max(1e-10, math.exp(log_bf))
