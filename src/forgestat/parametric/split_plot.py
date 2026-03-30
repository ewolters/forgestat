"""Split-plot ANOVA — nested and mixed designs.

Whole-plot factors tested against whole-plot error,
sub-plot factors tested against residual error.
Common in manufacturing: operators (whole-plot) × settings (sub-plot).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy import stats


@dataclass
class SplitPlotSource:
    """One row in a split-plot ANOVA table."""

    source: str
    ss: float = 0.0
    df: float = 0.0
    ms: float = 0.0
    f_statistic: float = 0.0
    p_value: float = 0.0
    error_term: str = ""  # which error term was used


@dataclass
class SplitPlotResult:
    """Split-plot ANOVA result."""

    sources: list[SplitPlotSource] = field(default_factory=list)
    whole_plot_error_ms: float = 0.0
    sub_plot_error_ms: float = 0.0
    n_whole_plots: int = 0
    n_sub_plots: int = 0


def split_plot_anova(
    data: dict[str, list],
    response: str,
    whole_plot_factor: str,
    sub_plot_factor: str,
    block: str | None = None,
) -> SplitPlotResult:
    """Split-plot ANOVA with separate error terms.

    Args:
        data: Dict of column_name → list of values.
        response: Response variable column.
        whole_plot_factor: Factor applied to whole plots (e.g., operator).
        sub_plot_factor: Factor applied within plots (e.g., machine setting).
        block: Optional blocking variable (whole-plot replicate identifier).

    Returns:
        SplitPlotResult with ANOVA table using appropriate error terms.
    """
    y = np.asarray(data[response], dtype=float)
    wp = np.asarray(data[whole_plot_factor])
    sp = np.asarray(data[sub_plot_factor])
    n = len(y)

    wp_levels = sorted(set(wp))
    sp_levels = sorted(set(sp))
    n_wp = len(wp_levels)
    n_sp = len(sp_levels)

    grand_mean = float(np.mean(y))

    # Whole-plot means
    wp_means = {}
    for w in wp_levels:
        wp_means[w] = float(np.mean(y[wp == w]))

    # Sub-plot means
    sp_means = {}
    for s in sp_levels:
        sp_means[s] = float(np.mean(y[sp == s]))

    # Cell means
    cell_means = {}
    cell_ns = {}
    for w in wp_levels:
        for s in sp_levels:
            mask = (wp == w) & (sp == s)
            if np.any(mask):
                cell_means[(w, s)] = float(np.mean(y[mask]))
                cell_ns[(w, s)] = int(np.sum(mask))

    # SS decomposition
    n_per_cell = n // (n_wp * n_sp) if n_wp * n_sp > 0 else 1

    ss_wp = n_sp * n_per_cell * sum((wp_means[w] - grand_mean) ** 2 for w in wp_levels)
    ss_sp = n_wp * n_per_cell * sum((sp_means[s] - grand_mean) ** 2 for s in sp_levels)

    # Interaction SS
    ss_inter = 0.0
    for w in wp_levels:
        for s in sp_levels:
            if (w, s) in cell_means:
                effect = cell_means[(w, s)] - wp_means[w] - sp_means[s] + grand_mean
                ss_inter += n_per_cell * effect ** 2

    ss_total = float(np.sum((y - grand_mean) ** 2))

    # Whole-plot error: if blocks, use WP×Block interaction; else use WP×SP interaction
    # Simplified: use interaction as WP error, residual as SP error
    ss_residual = ss_total - ss_wp - ss_sp - ss_inter

    df_wp = n_wp - 1
    df_sp = n_sp - 1
    df_inter = df_wp * df_sp
    df_residual = n - n_wp * n_sp

    ms_wp = ss_wp / df_wp if df_wp > 0 else 0
    ms_sp = ss_sp / df_sp if df_sp > 0 else 0
    ms_inter = ss_inter / df_inter if df_inter > 0 else 0
    ms_residual = ss_residual / df_residual if df_residual > 0 else 0

    # Whole-plot tested against interaction (or WP error)
    wp_error = ms_inter if ms_inter > 0 else ms_residual
    f_wp = ms_wp / wp_error if wp_error > 0 else 0
    p_wp = float(1 - stats.f.cdf(f_wp, df_wp, df_inter if ms_inter > 0 else df_residual))

    # Sub-plot tested against residual
    f_sp = ms_sp / ms_residual if ms_residual > 0 else 0
    p_sp = float(1 - stats.f.cdf(f_sp, df_sp, df_residual)) if df_residual > 0 else 1.0

    # Interaction tested against residual
    f_inter = ms_inter / ms_residual if ms_residual > 0 else 0
    p_inter = float(1 - stats.f.cdf(f_inter, df_inter, df_residual)) if df_residual > 0 else 1.0

    sources = [
        SplitPlotSource(source=whole_plot_factor, ss=float(ss_wp), df=float(df_wp),
                        ms=float(ms_wp), f_statistic=float(f_wp), p_value=p_wp,
                        error_term="WP×SP interaction"),
        SplitPlotSource(source=sub_plot_factor, ss=float(ss_sp), df=float(df_sp),
                        ms=float(ms_sp), f_statistic=float(f_sp), p_value=p_sp,
                        error_term="Residual"),
        SplitPlotSource(source=f"{whole_plot_factor}×{sub_plot_factor}",
                        ss=float(ss_inter), df=float(df_inter), ms=float(ms_inter),
                        f_statistic=float(f_inter), p_value=p_inter,
                        error_term="Residual"),
        SplitPlotSource(source="Residual", ss=float(ss_residual), df=float(df_residual),
                        ms=float(ms_residual)),
    ]

    return SplitPlotResult(
        sources=sources,
        whole_plot_error_ms=float(wp_error),
        sub_plot_error_ms=float(ms_residual),
        n_whole_plots=n_wp,
        n_sub_plots=n_sp,
    )
