"""Time series decomposition — classical and STL.

Trend, seasonal, and residual extraction.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from forgecore import ResultMixin


@dataclass
class DecompositionResult(ResultMixin):
    """Time series decomposition result."""

    model: str  # "additive" or "multiplicative"
    period: int
    trend: list[float] = field(default_factory=list)
    seasonal: list[float] = field(default_factory=list)
    residual: list[float] = field(default_factory=list)
    observed: list[float] = field(default_factory=list)
    seasonal_strength: float = 0.0  # 0-1
    trend_strength: float = 0.0  # 0-1
    trend_direction: str = ""  # "upward", "downward", "flat"
    trend_change: float = 0.0

    @property
    def summary(self) -> str:
        return (f"{self.model} decomposition, period {self.period}; "
                f"trend {self.trend_direction or 'n/a'} "
                f"(strength {self.trend_strength:.2f}), "
                f"seasonal strength {self.seasonal_strength:.2f}")

    def _panel(self, values: list[float], title: str):
        """A theme-neutral component line over period index.

        Trend/residual are undefined at the series boundaries (centered moving
        average), so plot only the defined points — a gap, not a crash.
        """
        from forgecore import ROLE_DATA, ChartSpec

        spec = ChartSpec(
            title=title, chart_type="line",
            x_axis={"label": "Period"}, y_axis={"label": title},
        )
        pts = [(i, v) for i, v in enumerate(values) if v is not None]
        spec.add_trace([i for i, _ in pts], [v for _, v in pts],
                       trace_type="line", color="", role=ROLE_DATA)
        return spec

    def to_render(self):
        """Primary portrait: the observed series."""
        return self._panel(self.observed, "Observed")

    def views(self) -> list:
        """Complete portrait: observed / trend / seasonal / residual panels."""
        panels = [("Observed", self.observed), ("Trend", self.trend),
                  ("Seasonal", self.seasonal), ("Residual", self.residual)]
        specs = [self._panel(vals, title) for title, vals in panels if vals]
        return specs or [self.to_render()]


def classical_decompose(
    data: list[float] | np.ndarray,
    period: int = 12,
    model: str = "additive",
) -> DecompositionResult:
    """Classical seasonal decomposition (moving average method).

    Args:
        data: Time series values.
        period: Seasonal period (e.g., 12 for monthly, 7 for daily).
        model: "additive" (Y = T + S + R) or "multiplicative" (Y = T × S × R).

    Returns:
        DecompositionResult with components and strength metrics.
    """
    from statsmodels.tsa.seasonal import seasonal_decompose

    x = np.asarray(data, dtype=float)

    if len(x) < 2 * period:
        raise ValueError(f"Need at least 2 full periods ({2 * period} points), got {len(x)}")

    decomp = seasonal_decompose(x, model=model, period=period)

    trend = np.array(decomp.trend)
    seasonal = np.array(decomp.seasonal)
    resid = np.array(decomp.resid)

    # Strengths (handling NaN in trend/resid from edge effects)
    resid_clean = resid[np.isfinite(resid)]
    seasonal_clean = seasonal[np.isfinite(seasonal)]
    trend_clean = trend[np.isfinite(trend)]

    if len(resid_clean) > 0 and len(seasonal_clean) > 0:
        s_var = float(np.var(seasonal_clean))
        r_var = float(np.var(resid_clean))
        seasonal_strength = max(0, 1 - r_var / (s_var + r_var)) if (s_var + r_var) > 0 else 0.0
    else:
        seasonal_strength = 0.0

    if len(resid_clean) > 0 and len(trend_clean) > 1:
        t_var = float(np.var(trend_clean))
        r_var = float(np.var(resid_clean))
        trend_strength = max(0, 1 - r_var / (t_var + r_var)) if (t_var + r_var) > 0 else 0.0
    else:
        trend_strength = 0.0

    # Trend direction
    if len(trend_clean) > 1:
        t_start = float(trend_clean[0])
        t_end = float(trend_clean[-1])
        t_change = t_end - t_start
        if abs(t_change) < 0.01 * abs(t_start) if t_start != 0 else abs(t_change) < 0.01:
            t_dir = "flat"
        elif t_change > 0:
            t_dir = "upward"
        else:
            t_dir = "downward"
    else:
        t_dir = "flat"
        t_change = 0.0

    return DecompositionResult(
        model=model,
        period=period,
        trend=[float(v) if np.isfinite(v) else None for v in trend],
        seasonal=[float(v) for v in seasonal],
        residual=[float(v) if np.isfinite(v) else None for v in resid],
        observed=[float(v) for v in x],
        seasonal_strength=float(seasonal_strength),
        trend_strength=float(trend_strength),
        trend_direction=t_dir,
        trend_change=float(t_change),
    )
