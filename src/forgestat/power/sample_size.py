"""Power analysis and sample size calculations for all test types.

Pure scipy — no external dependencies beyond numpy/scipy.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

from scipy import stats


@dataclass
class PowerResult:
    """Power analysis result."""

    test: str
    power: float = 0.0
    sample_size: int = 0
    alpha: float = 0.05
    effect_size: float = 0.0
    detail: dict = field(default_factory=dict)


def power_t_test(
    effect_size: float,
    n: int | None = None,
    alpha: float = 0.05,
    power: float | None = None,
    alternative: str = "two-sided",
    test_type: str = "one_sample",
) -> PowerResult:
    """Power or sample size for t-tests.

    Provide either n (to compute power) or power (to compute n).

    Args:
        effect_size: Cohen's d.
        n: Sample size per group (if computing power).
        alpha: Significance level.
        power: Target power (if computing sample size).
        alternative: "two-sided" or "one-sided".
        test_type: "one_sample", "two_sample", or "paired".
    """
    sides = 2 if alternative == "two-sided" else 1
    d = abs(effect_size)

    if n is not None:
        # Compute power
        if test_type == "two_sample":
            df = 2 * n - 2
            ncp = d * math.sqrt(n / 2)
        else:  # one_sample or paired
            df = n - 1
            ncp = d * math.sqrt(n)

        t_crit = stats.t.ppf(1 - alpha / sides, df)
        pw = 1 - stats.nct.cdf(t_crit, df, ncp)
        if sides == 2:
            pw += stats.nct.cdf(-t_crit, df, ncp)

        return PowerResult(test=f"t_test_{test_type}", power=float(pw), sample_size=n,
                           alpha=alpha, effect_size=d)

    elif power is not None:
        # Compute sample size via search
        for n_try in range(2, 10000):
            result = power_t_test(d, n=n_try, alpha=alpha, alternative=alternative, test_type=test_type)
            if result.power >= power:
                return PowerResult(test=f"t_test_{test_type}", power=result.power, sample_size=n_try,
                                   alpha=alpha, effect_size=d)

        return PowerResult(test=f"t_test_{test_type}", power=0.0, sample_size=10000,
                           alpha=alpha, effect_size=d, detail={"note": "max iterations reached"})

    raise ValueError("Provide either n (for power) or power (for sample size)")


def power_anova(
    effect_size: float,
    k: int,
    n_per_group: int | None = None,
    alpha: float = 0.05,
    power: float | None = None,
) -> PowerResult:
    """Power or sample size for one-way ANOVA.

    Args:
        effect_size: Cohen's f = sqrt(η² / (1 - η²)).
        k: Number of groups.
        n_per_group: Sample size per group (if computing power).
        alpha: Significance level.
        power: Target power (if computing sample size).
    """
    f = abs(effect_size)

    if n_per_group is not None:
        N = k * n_per_group
        df1 = k - 1
        df2 = N - k
        ncp = f ** 2 * N
        f_crit = stats.f.ppf(1 - alpha, df1, df2)
        pw = 1 - stats.ncf.cdf(f_crit, df1, df2, ncp)

        return PowerResult(test="anova", power=float(pw), sample_size=n_per_group,
                           alpha=alpha, effect_size=f, detail={"k": k, "total_n": N})

    elif power is not None:
        for n_try in range(2, 5000):
            result = power_anova(f, k, n_per_group=n_try, alpha=alpha)
            if result.power >= power:
                return PowerResult(test="anova", power=result.power, sample_size=n_try,
                                   alpha=alpha, effect_size=f, detail={"k": k, "total_n": k * n_try})

        return PowerResult(test="anova", power=0.0, sample_size=5000,
                           alpha=alpha, effect_size=f, detail={"note": "max iterations reached"})

    raise ValueError("Provide either n_per_group or power")


def power_proportion(
    p1: float,
    p2: float | None = None,
    p0: float | None = None,
    n: int | None = None,
    alpha: float = 0.05,
    power: float | None = None,
) -> PowerResult:
    """Power or sample size for proportion test.

    For 1-proportion: provide p1 (true) and p0 (hypothesized).
    For 2-proportions: provide p1 and p2.

    Args:
        p1: True/group-1 proportion.
        p2: Group-2 proportion (for 2-sample).
        p0: Hypothesized proportion (for 1-sample).
        n: Sample size (if computing power).
        alpha: Significance level.
        power: Target power (if computing sample size).
    """
    if p0 is not None:
        # 1-proportion
        h = 2 * abs(math.asin(math.sqrt(p1)) - math.asin(math.sqrt(p0)))
    elif p2 is not None:
        # 2-proportions
        h = 2 * abs(math.asin(math.sqrt(p1)) - math.asin(math.sqrt(p2)))
    else:
        raise ValueError("Provide p0 (1-sample) or p2 (2-sample)")

    z_a = stats.norm.ppf(1 - alpha / 2)

    if n is not None:
        ncp = h * math.sqrt(n)
        pw = 1 - stats.norm.cdf(z_a - ncp) + stats.norm.cdf(-z_a - ncp)
        return PowerResult(test="proportion", power=float(pw), sample_size=n,
                           alpha=alpha, effect_size=h)

    elif power is not None:
        z_b = stats.norm.ppf(power)
        n_est = math.ceil(((z_a + z_b) / h) ** 2) if h > 0 else 10000
        # Verify
        result = power_proportion(p1, p2=p2, p0=p0, n=n_est, alpha=alpha)
        while result.power < power and n_est < 100000:
            n_est += 1
            result = power_proportion(p1, p2=p2, p0=p0, n=n_est, alpha=alpha)
        return PowerResult(test="proportion", power=result.power, sample_size=n_est,
                           alpha=alpha, effect_size=h)

    raise ValueError("Provide either n or power")


def power_chi_square(
    effect_size: float,
    df: int,
    n: int | None = None,
    alpha: float = 0.05,
    power: float | None = None,
) -> PowerResult:
    """Power or sample size for chi-square test.

    Args:
        effect_size: Cohen's w.
        df: Degrees of freedom.
        n: Sample size (if computing power).
        alpha: Significance level.
        power: Target power (if computing sample size).
    """
    w = abs(effect_size)

    if n is not None:
        ncp = w ** 2 * n
        chi2_crit = stats.chi2.ppf(1 - alpha, df)
        pw = 1 - stats.ncx2.cdf(chi2_crit, df, ncp)
        return PowerResult(test="chi_square", power=float(pw), sample_size=n,
                           alpha=alpha, effect_size=w, detail={"df": df})

    elif power is not None:
        for n_try in range(10, 100000, 5):
            result = power_chi_square(w, df, n=n_try, alpha=alpha)
            if result.power >= power:
                return PowerResult(test="chi_square", power=result.power, sample_size=n_try,
                                   alpha=alpha, effect_size=w, detail={"df": df})

        return PowerResult(test="chi_square", power=0.0, sample_size=100000,
                           alpha=alpha, effect_size=w, detail={"note": "max reached"})

    raise ValueError("Provide either n or power")


def sample_size_for_ci(
    target_width: float,
    std: float | None = None,
    proportion: float | None = None,
    conf: float = 0.95,
) -> int:
    """Sample size to achieve a target CI half-width.

    For means: provide std. For proportions: provide proportion.

    Args:
        target_width: Desired CI half-width (margin of error).
        std: Population standard deviation (for mean CI).
        proportion: Estimated proportion (for proportion CI).
        conf: Confidence level.

    Returns:
        Required sample size.
    """
    z = stats.norm.ppf((1 + conf) / 2)

    if std is not None:
        n = math.ceil((z * std / target_width) ** 2)
    elif proportion is not None:
        p = proportion
        n = math.ceil(z ** 2 * p * (1 - p) / target_width ** 2)
    else:
        raise ValueError("Provide std (for mean) or proportion")

    return max(2, n)


def power_z_test(
    effect_size: float,
    n: int | None = None,
    alpha: float = 0.05,
    power: float | None = None,
) -> PowerResult:
    """Power for one-sample Z-test (known sigma, large n).

    Equivalent to power_t_test for large samples. Provided for API completeness.
    """
    # Z-test is t-test in the large-n limit
    return power_t_test(effect_size, n=n, alpha=alpha, power=power, test_type="one_sample")


def power_equivalence(
    effect_size: float,
    margin: float,
    n: int | None = None,
    alpha: float = 0.05,
    power: float | None = None,
) -> PowerResult:
    """Power for TOST equivalence test.

    Args:
        effect_size: True difference (Cohen's d scale).
        margin: Equivalence margin (Cohen's d scale).
        n: Sample size per group.
        alpha: Significance level.
        power: Target power (for sample size computation).
    """
    z_a = stats.norm.ppf(1 - alpha)

    if n is not None:
        se = 1 / math.sqrt(n / 2)
        # Power ≈ Φ((margin - |δ|) / se - z_α)
        pw = float(stats.norm.cdf((margin - abs(effect_size)) / se - z_a))
        return PowerResult(test="equivalence", power=pw, sample_size=n,
                           alpha=alpha, effect_size=effect_size, detail={"margin": margin})

    elif power is not None:
        for n_try in range(4, 10000):
            result = power_equivalence(effect_size, margin, n=n_try, alpha=alpha)
            if result.power >= power:
                return PowerResult(test="equivalence", power=result.power, sample_size=n_try,
                                   alpha=alpha, effect_size=effect_size, detail={"margin": margin})
        return PowerResult(test="equivalence", power=0.0, sample_size=10000,
                           alpha=alpha, effect_size=effect_size)

    raise ValueError("Provide either n or power")


def sample_size_tolerance(
    coverage: float = 0.95,
    confidence: float = 0.95,
    std: float = 1.0,
    target_width: float | None = None,
) -> int:
    """Sample size for a tolerance interval.

    Finds n such that the normal tolerance interval k-factor achieves
    the desired coverage at the given confidence level.

    Args:
        coverage: Proportion of population to cover.
        confidence: Confidence level.
        std: Estimated population std (used if target_width specified).
        target_width: Desired half-width of tolerance interval.

    Returns:
        Required sample size.
    """
    z_p = stats.norm.ppf((1 + coverage) / 2)

    # Search for n where k-factor ≈ z_p (within 10% margin)
    for n in range(3, 10000):
        chi2_val = stats.chi2.ppf(1 - confidence, n - 1)
        if chi2_val <= 0:
            continue
        k = z_p * math.sqrt((n - 1) / chi2_val) * math.sqrt(1 + 1 / n)

        if target_width is not None:
            # Check if k * std ≤ target_width
            if k * std <= target_width:
                return n
        else:
            # Check if k is close to z_p (within 10%)
            if k < z_p * 1.10:
                return n

    return 10000
