"""Shared distribution views — theme-neutral forgecore ChartSpecs built from
raw samples a result owns (§5b). Group tests draw a box plot + a per-group
normal Q-Q; one-sample tests draw a histogram + a Q-Q. No forgeviz import:
the result self-renders through the contract fallback.
"""

from __future__ import annotations

import numpy as np

from forgecore import ROLE_CONTROL_LIMIT, ROLE_DATA, ChartSpec


def _box_spec(groups: dict[str, list[float]], title: str) -> ChartSpec:
    """Box plot across named groups — dict traces the SVG renderer reads.
    Color is omitted so the renderer resolves it from the theme palette.
    """
    spec = ChartSpec(title=title, chart_type="box_plot",
                     x_axis={"label": ""}, y_axis={"label": "Value"})
    for i, (name, values) in enumerate(groups.items()):
        vals = [float(v) for v in values]
        if not vals:
            continue
        q1, median, q3 = (float(np.percentile(vals, p)) for p in (25, 50, 75))
        iqr = q3 - q1
        wl = max(min(vals), q1 - 1.5 * iqr)
        wh = min(max(vals), q3 + 1.5 * iqr)
        spec.traces.append({
            "type": "box", "name": str(name), "q1": q1, "median": median,
            "q3": q3, "whisker_low": wl, "whisker_high": wh,
            "outliers": [v for v in vals if v < wl or v > wh], "x_position": i,
        })
    return spec


def _qq_spec(values: list[float], title: str) -> ChartSpec:
    """Normal Q-Q: ordered values vs theoretical normal quantiles (Blom
    positions) with a mean±std reference line. Surfaces the normality
    assumption the parametric tests rest on.
    """
    from scipy import stats

    ordered = sorted(float(v) for v in values)
    n = len(ordered)
    theoretical = [float(stats.norm.ppf((i + 1 - 0.375) / (n + 0.25)))
                   for i in range(n)]
    spec = ChartSpec(title=title, chart_type="qq_plot",
                     x_axis={"label": "Normal Quantile"},
                     y_axis={"label": "Ordered Value"})
    spec.add_trace(theoretical, ordered, trace_type="scatter", color="",
                   role=ROLE_DATA)
    mean = float(np.mean(ordered))
    std = float(np.std(ordered, ddof=1)) if n > 1 else 0.0
    ends = [theoretical[0], theoretical[-1]]
    spec.add_trace(ends, [mean + std * z for z in ends], trace_type="line",
                   dash="dashed", color="", role=ROLE_CONTROL_LIMIT)
    return spec


def _hist_spec(values: list[float], title: str) -> ChartSpec:
    """Histogram of one sample — binned counts as a bar chart, theme-neutral."""
    vals = [float(v) for v in values]
    counts, edges = np.histogram(vals, bins=min(20, max(5, len(vals) // 2)))
    centers = [f"{(edges[i] + edges[i + 1]) / 2:.3g}" for i in range(len(counts))]
    spec = ChartSpec(title=title, chart_type="histogram",
                     x_axis={"label": "Value"}, y_axis={"label": "Frequency"})
    spec.add_trace(centers, counts.tolist(), trace_type="bar", color="",
                   role=ROLE_DATA)
    return spec


def box_views(groups: dict[str, list[float]], title: str = "Group Comparison") -> list[ChartSpec]:
    """A box plot of the groups + a normal Q-Q for each group with >= 3 points."""
    present = {k: v for k, v in groups.items() if v}
    views = [_box_spec(present, title)]
    for name, vals in present.items():
        if len(vals) >= 3:
            views.append(_qq_spec(vals, f"Normal Q-Q — {name}"))
    return views


def histogram_views(values: list[float], title: str = "Distribution") -> list[ChartSpec]:
    """A histogram of one sample + a normal Q-Q (>= 3 points)."""
    views = [_hist_spec(values, title)]
    if len(values) >= 3:
        views.append(_qq_spec(values, "Normal Q-Q Plot"))
    return views


def sample_views(samples: dict[str, list[float]], title: str = "Distribution") -> list[ChartSpec]:
    """Shape by sample count: >= 2 groups -> box plot; one sample -> histogram."""
    present = {k: v for k, v in samples.items() if v}
    if len(present) >= 2:
        return box_views(present, title)
    if len(present) == 1:
        return histogram_views(next(iter(present.values())), title)
    return [ChartSpec(title=title, chart_type="histogram")]
