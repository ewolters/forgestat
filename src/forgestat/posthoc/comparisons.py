"""Post-hoc multiple comparison procedures.

Tukey HSD, Dunnett, Games-Howell, Scheffé, Bonferroni, Dunn.
"""

from __future__ import annotations

import math
from itertools import combinations

import numpy as np
from scipy import stats

from ..core.types import PostHocComparison, PostHocResult


def tukey_hsd(
    *groups: list[float] | np.ndarray,
    labels: list[str] | None = None,
    alpha: float = 0.05,
) -> PostHocResult:
    """Tukey's Honest Significant Difference — all pairwise comparisons.

    Controls family-wise error rate using the Studentized Range distribution.

    Args:
        *groups: Two or more groups.
        labels: Group names.
        alpha: Significance level.
    """
    arrays = [np.asarray(g, dtype=float) for g in groups]
    arrays = [a[np.isfinite(a)] for a in arrays]
    names = labels or [f"Group {i+1}" for i in range(len(arrays))]
    k = len(arrays)

    if k < 2:
        raise ValueError("Need at least 2 groups for post-hoc test")

    # Pool within-group variance
    all_n = [len(a) for a in arrays]
    n_total = sum(all_n)
    df_within = n_total - k
    ms_within = sum((len(a) - 1) * float(np.var(a, ddof=1)) for a in arrays) / df_within if df_within > 0 else 0

    group_means = {name: float(np.mean(a)) for name, a in zip(names, arrays)}
    comparisons = []

    for (i, name_i), (j, name_j) in combinations(enumerate(names), 2):
        ni, nj = all_n[i], all_n[j]
        mean_diff = float(np.mean(arrays[i]) - np.mean(arrays[j]))
        se = math.sqrt(ms_within * (1 / ni + 1 / nj) / 2) if ms_within > 0 else 0

        q = abs(mean_diff) / se if se > 0 else 0.0

        # p-value from Studentized Range distribution
        try:
            p = float(stats.studentized_range.sf(q, k, df_within))
        except Exception:
            # Fallback: Bonferroni-adjusted t
            t_val = abs(mean_diff) / (math.sqrt(ms_within * (1 / ni + 1 / nj)) if ms_within > 0 else 1)
            p_raw = 2 * (1 - stats.t.cdf(t_val, df_within))
            n_pairs = k * (k - 1) / 2
            p = min(1.0, p_raw * n_pairs)

        # CI
        try:
            q_crit = stats.studentized_range.ppf(1 - alpha, k, df_within)
        except Exception:
            q_crit = stats.t.ppf(1 - alpha / (k * (k - 1)), df_within) * math.sqrt(2)

        hw = q_crit * se
        ci_lo = mean_diff - hw
        ci_hi = mean_diff + hw

        comparisons.append(PostHocComparison(
            group1=name_i,
            group2=name_j,
            mean_diff=mean_diff,
            se=se,
            t_or_q=q,
            p_value=p,
            ci_lower=ci_lo,
            ci_upper=ci_hi,
            significant=p < alpha,
            reject=p < alpha,
        ))

    return PostHocResult(
        test_name="tukey_hsd",
        comparisons=comparisons,
        alpha=alpha,
        correction="studentized_range",
        group_means=group_means,
    )


def games_howell(
    *groups: list[float] | np.ndarray,
    labels: list[str] | None = None,
    alpha: float = 0.05,
) -> PostHocResult:
    """Games-Howell test — post-hoc for unequal variances.

    Does NOT assume equal variances. Uses Welch-Satterthwaite df.

    Args:
        *groups: Two or more groups.
        labels: Group names.
        alpha: Significance level.
    """
    arrays = [np.asarray(g, dtype=float) for g in groups]
    arrays = [a[np.isfinite(a)] for a in arrays]
    names = labels or [f"Group {i+1}" for i in range(len(arrays))]
    k = len(arrays)

    group_means = {name: float(np.mean(a)) for name, a in zip(names, arrays)}
    comparisons = []

    for (i, name_i), (j, name_j) in combinations(enumerate(names), 2):
        ni, nj = len(arrays[i]), len(arrays[j])
        si2 = float(np.var(arrays[i], ddof=1))
        sj2 = float(np.var(arrays[j], ddof=1))

        mean_diff = float(np.mean(arrays[i]) - np.mean(arrays[j]))
        se = math.sqrt(si2 / ni + sj2 / nj)

        # Welch-Satterthwaite df
        if se > 0:
            num = (si2 / ni + sj2 / nj) ** 2
            den = (si2 / ni) ** 2 / (ni - 1) + (sj2 / nj) ** 2 / (nj - 1)
            df = num / den if den > 0 else ni + nj - 2
        else:
            df = ni + nj - 2

        q = abs(mean_diff) / se if se > 0 else 0.0

        # p-value from Studentized Range
        try:
            p = float(stats.studentized_range.sf(q, k, df))
        except Exception:
            t_val = q / math.sqrt(2)
            p_raw = 2 * (1 - stats.t.cdf(t_val, df))
            p = min(1.0, p_raw * k * (k - 1) / 2)

        try:
            q_crit = stats.studentized_range.ppf(1 - alpha, k, df)
        except Exception:
            q_crit = 3.0  # conservative fallback

        hw = q_crit * se
        ci_lo = mean_diff - hw
        ci_hi = mean_diff + hw

        comparisons.append(PostHocComparison(
            group1=name_i,
            group2=name_j,
            mean_diff=mean_diff,
            se=se,
            t_or_q=q,
            p_value=p,
            ci_lower=ci_lo,
            ci_upper=ci_hi,
            significant=p < alpha,
            reject=p < alpha,
        ))

    return PostHocResult(
        test_name="games_howell",
        comparisons=comparisons,
        alpha=alpha,
        correction="welch_studentized_range",
        group_means=group_means,
    )


def dunnett(
    control: list[float] | np.ndarray,
    *treatments: list[float] | np.ndarray,
    control_name: str = "Control",
    treatment_names: list[str] | None = None,
    alpha: float = 0.05,
) -> PostHocResult:
    """Dunnett's test — compare each treatment to a single control.

    More powerful than Tukey when only control-vs-treatment comparisons are needed.

    Args:
        control: Control group data.
        *treatments: Treatment group data.
        control_name: Name of control group.
        treatment_names: Names of treatment groups.
        alpha: Significance level.
    """
    ctrl = np.asarray(control, dtype=float)
    ctrl = ctrl[np.isfinite(ctrl)]
    treat_arrays = [np.asarray(t, dtype=float) for t in treatments]
    treat_arrays = [a[np.isfinite(a)] for a in treat_arrays]
    t_names = treatment_names or [f"Treatment {i+1}" for i in range(len(treat_arrays))]
    k = len(treat_arrays) + 1  # total groups

    # Try scipy.stats.dunnett (available in scipy >= 1.11)
    try:
        result = stats.dunnett(*treat_arrays, control=ctrl)
        comparisons = []
        for i, name in enumerate(t_names):
            comparisons.append(PostHocComparison(
                group1=name,
                group2=control_name,
                mean_diff=float(np.mean(treat_arrays[i]) - np.mean(ctrl)),
                p_value=float(result.pvalue[i]),
                significant=float(result.pvalue[i]) < alpha,
                reject=float(result.pvalue[i]) < alpha,
            ))
        group_means = {control_name: float(np.mean(ctrl))}
        group_means.update({name: float(np.mean(a)) for name, a in zip(t_names, treat_arrays)})

        return PostHocResult(
            test_name="dunnett",
            comparisons=comparisons,
            alpha=alpha,
            correction="dunnett_distribution",
            group_means=group_means,
            control_group=control_name,
        )
    except (AttributeError, TypeError):
        pass

    # Fallback: Welch t-tests with Bonferroni correction
    n_comparisons = len(treat_arrays)
    all_groups = [ctrl] + treat_arrays
    all_n = [len(a) for a in all_groups]
    n_total = sum(all_n)
    df_within = n_total - k
    ms_within = sum((len(a) - 1) * float(np.var(a, ddof=1)) for a in all_groups) / df_within if df_within > 0 else 0

    group_means = {control_name: float(np.mean(ctrl))}
    comparisons = []

    for i, (name, arr) in enumerate(zip(t_names, treat_arrays)):
        ni = len(arr)
        nc = len(ctrl)
        mean_diff = float(np.mean(arr) - np.mean(ctrl))
        se = math.sqrt(ms_within * (1 / ni + 1 / nc)) if ms_within > 0 else 0
        t_val = abs(mean_diff) / se if se > 0 else 0
        p_raw = 2 * (1 - stats.t.cdf(t_val, df_within))
        p_adj = min(1.0, p_raw * n_comparisons)

        group_means[name] = float(np.mean(arr))
        comparisons.append(PostHocComparison(
            group1=name,
            group2=control_name,
            mean_diff=mean_diff,
            se=se,
            t_or_q=t_val,
            p_value=p_adj,
            significant=p_adj < alpha,
            reject=p_adj < alpha,
        ))

    return PostHocResult(
        test_name="dunnett",
        comparisons=comparisons,
        alpha=alpha,
        correction="bonferroni",
        group_means=group_means,
        control_group=control_name,
    )


def dunn(
    *groups: list[float] | np.ndarray,
    labels: list[str] | None = None,
    alpha: float = 0.05,
) -> PostHocResult:
    """Dunn's test — non-parametric post-hoc after Kruskal-Wallis.

    Rank-based pairwise comparisons with Bonferroni correction.

    Args:
        *groups: Two or more groups (same as passed to kruskal_wallis).
        labels: Group names.
        alpha: Significance level.
    """
    arrays = [np.asarray(g, dtype=float) for g in groups]
    arrays = [a[np.isfinite(a)] for a in arrays]
    names = labels or [f"Group {i+1}" for i in range(len(arrays))]
    k = len(arrays)

    # Compute ranks across all data
    all_data = np.concatenate(arrays)
    ranks = stats.rankdata(all_data)
    n_total = len(all_data)

    # Mean rank per group
    cumulative = 0
    mean_ranks = {}
    group_ns = {}
    for name, arr in zip(names, arrays):
        n_g = len(arr)
        group_ranks = ranks[cumulative:cumulative + n_g]
        mean_ranks[name] = float(np.mean(group_ranks))
        group_ns[name] = n_g
        cumulative += n_g

    # Tied-rank correction
    _, tie_counts = np.unique(ranks, return_counts=True)
    tie_correction = 1.0 - np.sum(tie_counts ** 3 - tie_counts) / (n_total ** 3 - n_total) if n_total > 1 else 1.0

    n_pairs = k * (k - 1) // 2
    group_means = {name: float(np.mean(a)) for name, a in zip(names, arrays)}
    comparisons = []

    for (i, name_i), (j, name_j) in combinations(enumerate(names), 2):
        ni = group_ns[name_i]
        nj = group_ns[name_j]
        diff_rank = mean_ranks[name_i] - mean_ranks[name_j]
        se = math.sqrt(tie_correction * n_total * (n_total + 1) / 12 * (1 / ni + 1 / nj))
        z = diff_rank / se if se > 0 else 0.0

        p_raw = 2 * (1 - stats.norm.cdf(abs(z)))
        p_adj = min(1.0, p_raw * n_pairs)

        comparisons.append(PostHocComparison(
            group1=name_i,
            group2=name_j,
            mean_diff=float(np.mean(arrays[i]) - np.mean(arrays[j])),
            se=se,
            t_or_q=z,
            p_value=p_adj,
            significant=p_adj < alpha,
            reject=p_adj < alpha,
        ))

    return PostHocResult(
        test_name="dunn",
        comparisons=comparisons,
        alpha=alpha,
        correction="bonferroni",
        group_means=group_means,
    )


def bonferroni(
    *groups: list[float] | np.ndarray,
    labels: list[str] | None = None,
    alpha: float = 0.05,
) -> PostHocResult:
    """Bonferroni-corrected pairwise t-tests.

    Simple correction: multiply p-values by number of comparisons.

    Args:
        *groups: Two or more groups.
        labels: Group names.
        alpha: Significance level.
    """
    arrays = [np.asarray(g, dtype=float) for g in groups]
    arrays = [a[np.isfinite(a)] for a in arrays]
    names = labels or [f"Group {i+1}" for i in range(len(arrays))]
    k = len(arrays)
    n_pairs = k * (k - 1) // 2

    group_means = {name: float(np.mean(a)) for name, a in zip(names, arrays)}
    comparisons = []

    for (i, name_i), (j, name_j) in combinations(enumerate(names), 2):
        t_stat, p_raw = stats.ttest_ind(arrays[i], arrays[j], equal_var=False)
        p_adj = min(1.0, float(p_raw) * n_pairs)
        mean_diff = float(np.mean(arrays[i]) - np.mean(arrays[j]))

        comparisons.append(PostHocComparison(
            group1=name_i,
            group2=name_j,
            mean_diff=mean_diff,
            t_or_q=float(t_stat),
            p_value=p_adj,
            significant=p_adj < alpha,
            reject=p_adj < alpha,
        ))

    return PostHocResult(
        test_name="bonferroni",
        comparisons=comparisons,
        alpha=alpha,
        correction="bonferroni",
        group_means=group_means,
    )


def scheffe(
    *groups: list[float] | np.ndarray,
    labels: list[str] | None = None,
    alpha: float = 0.05,
) -> PostHocResult:
    """Scheffé's test — most conservative omnibus post-hoc.

    Controls family-wise error for ALL possible contrasts (not just pairwise).
    Uses the F-distribution critical value scaled by (k-1).

    Args:
        *groups: Two or more groups.
        labels: Group names.
        alpha: Significance level.
    """
    arrays = [np.asarray(g, dtype=float) for g in groups]
    arrays = [a[np.isfinite(a)] for a in arrays]
    names = labels or [f"Group {i+1}" for i in range(len(arrays))]
    k = len(arrays)

    all_n = [len(a) for a in arrays]
    n_total = sum(all_n)
    df_within = n_total - k
    ms_within = sum((len(a) - 1) * float(np.var(a, ddof=1)) for a in arrays) / df_within if df_within > 0 else 0

    group_means = {name: float(np.mean(a)) for name, a in zip(names, arrays)}
    comparisons = []

    for (i, name_i), (j, name_j) in combinations(enumerate(names), 2):
        ni, nj = all_n[i], all_n[j]
        mean_diff = float(np.mean(arrays[i]) - np.mean(arrays[j]))
        se = math.sqrt(ms_within * (1 / ni + 1 / nj)) if ms_within > 0 else 0

        f_val = (mean_diff ** 2) / (ms_within * (1 / ni + 1 / nj)) if se > 0 else 0

        # p-value from F distribution scaled by (k-1)
        p_val = float(1 - stats.f.cdf(f_val / (k - 1), k - 1, df_within)) if k > 1 and df_within > 0 else 1.0

        comparisons.append(PostHocComparison(
            group1=name_i,
            group2=name_j,
            mean_diff=mean_diff,
            se=se,
            t_or_q=math.sqrt(f_val) if f_val > 0 else 0,
            p_value=p_val,
            significant=p_val < alpha,
            reject=p_val < alpha,
        ))

    return PostHocResult(
        test_name="scheffe",
        comparisons=comparisons,
        alpha=alpha,
        correction="scheffe_f",
        group_means=group_means,
    )
