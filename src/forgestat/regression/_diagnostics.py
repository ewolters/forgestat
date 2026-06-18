"""Shared residual diagnostics — the 4-in-1 panel built from fitted/residuals.

RegressionResult and NonlinearResult both carry fitted + residuals; this is the
single place that turns them into the four standard diagnostic ChartSpecs, so
each result self-renders its panel via the engine contract (theme-neutral
structure only — no colors, role-tagged).
"""

from __future__ import annotations

import numpy as np
from scipy import stats

from forgecore import ROLE_CENTERLINE, ROLE_CONTROL_LIMIT, ROLE_DATA, ChartSpec


def residual_vs_fitted(fitted, residuals) -> ChartSpec:
    spec = ChartSpec(title="Residuals vs Fitted", chart_type="scatter",
                     x_axis={"label": "Fitted Value"}, y_axis={"label": "Residual"})
    spec.add_trace(list(fitted), list(residuals), trace_type="scatter",
                   color="", role=ROLE_DATA)
    spec.add_reference_line(0, axis="y", color="", role=ROLE_CENTERLINE)
    return spec


def _qq(residuals) -> ChartSpec:
    (osm, osr), (slope, intercept, _r) = stats.probplot(residuals, dist="norm")
    spec = ChartSpec(title="Normal Q-Q", chart_type="scatter",
                     x_axis={"label": "Theoretical Quantiles"},
                     y_axis={"label": "Ordered Residuals"})
    spec.add_trace(osm.tolist(), osr.tolist(), trace_type="scatter",
                   color="", role=ROLE_DATA)
    line_x = [float(osm.min()), float(osm.max())]
    spec.add_trace(line_x, [slope * x + intercept for x in line_x],
                   trace_type="line", dash="dashed", color="", role=ROLE_CONTROL_LIMIT)
    return spec


def _residual_histogram(residuals) -> ChartSpec:
    counts, edges = np.histogram(residuals, bins=15)
    centers = [f"{(edges[i] + edges[i + 1]) / 2:.2g}" for i in range(len(counts))]
    spec = ChartSpec(title="Residual Distribution", chart_type="bar",
                     x_axis={"label": "Residual"}, y_axis={"label": "Frequency"})
    spec.add_trace(centers, counts.tolist(), trace_type="bar", color="", role=ROLE_DATA)
    return spec


def _residual_vs_order(residuals) -> ChartSpec:
    spec = ChartSpec(title="Residuals vs Run Order", chart_type="line",
                     x_axis={"label": "Run Order"}, y_axis={"label": "Residual"})
    spec.add_trace(list(range(1, len(residuals) + 1)), list(residuals),
                   trace_type="line", color="", role=ROLE_DATA)
    spec.add_reference_line(0, axis="y", color="", role=ROLE_CENTERLINE)
    return spec


def residual_diagnostics(fitted, residuals) -> list:
    """The 4-in-1 residual diagnostic ChartSpecs, or [] if there's no data."""
    if not list(fitted) or not list(residuals):
        return []
    return [residual_vs_fitted(fitted, residuals), _qq(residuals),
            _residual_histogram(residuals), _residual_vs_order(residuals)]
