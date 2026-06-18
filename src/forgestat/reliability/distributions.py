"""Reliability distributions — Weibull, lognormal, exponential fitting.

MLE fitting, parameter estimation, reliability/hazard functions.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
from forgecore import ResultMixin
from scipy import stats


@dataclass
class WeibullFit(ResultMixin):
    """Weibull distribution fit result."""

    shape: float = 0.0  # β (shape/slope)
    scale: float = 0.0  # η (characteristic life)
    location: float = 0.0  # γ (location/threshold)
    b10_life: float = 0.0  # 10% failure life
    mean_life: float = 0.0  # MTTF
    median_life: float = 0.0
    ks_statistic: float = 0.0
    ks_p_value: float = 0.0
    failure_mode: str = ""  # "infant_mortality", "random", "wear_out"
    failure_times: list[float] = field(default_factory=list)  # raw sample — views() draw from it (§5b)

    @property
    def summary(self) -> str:
        return (f"Weibull beta={self.shape:.2f}, eta={self.scale:.1f}; "
                f"{self.failure_mode or 'n/a'} (B10={self.b10_life:.1f})")

    def _probability_plot(self):
        """Linearized Weibull CDF: ln(t) vs ln(-ln(1-F)) + the fitted line."""
        from forgecore import ROLE_CONTROL_LIMIT, ROLE_DATA, ChartSpec

        spec = ChartSpec(title="Weibull Probability Plot", chart_type="scatter",
                         x_axis={"label": "ln(Time)"}, y_axis={"label": "ln(-ln(1-F))"})
        times = sorted(t for t in self.failure_times if t > 0)
        if not times:
            return spec
        n = len(times)
        ranks = [(i - 0.3) / (n + 0.4) for i in range(1, n + 1)]  # Bernard's median rank
        x = [math.log(t) for t in times]
        y = [math.log(-math.log(1 - r)) if r < 1 else 5.0 for r in ranks]
        spec.add_trace(x, y, name="Data", trace_type="scatter", color="", role=ROLE_DATA)
        if self.scale > 0:
            ends = [min(x), max(x)]
            spec.add_trace(ends, [self.shape * (xv - math.log(self.scale)) for xv in ends],
                           name="Fit", trace_type="line", dash="dashed", color="",
                           role=ROLE_CONTROL_LIMIT)
        return spec

    def _survival_curve(self):
        """Empirical product-limit survival from the failure sample."""
        from forgecore import ROLE_CENTERLINE, ROLE_DATA, ChartSpec

        spec = ChartSpec(title="Survival Curve", chart_type="line",
                         x_axis={"label": "Time"}, y_axis={"label": "Survival Probability"})
        times = sorted(self.failure_times)
        if not times:
            return spec
        at_risk, s = len(times), 1.0
        xs, ys = [0.0], [1.0]
        for t in times:
            s *= (at_risk - 1) / at_risk if at_risk > 0 else 0.0
            xs.append(t)
            ys.append(s)
            at_risk -= 1
        xs.append(times[-1] * 1.1)
        ys.append(s)
        spec.add_trace(xs, ys, name="Survival", trace_type="step", color="", role=ROLE_DATA)
        spec.add_reference_line(0.5, axis="y", dash="dotted", color="", role=ROLE_CENTERLINE)
        return spec

    def _hazard(self):
        """Bathtub hazard h(t) = (beta/eta)*(t/eta)^(beta-1) from the parameters."""
        from forgecore import ROLE_DATA, ChartSpec

        spec = ChartSpec(title="Hazard Function", chart_type="line",
                         x_axis={"label": "Time"}, y_axis={"label": "Hazard Rate h(t)"})
        max_time = self.scale * 3 if self.scale > 0 else 1.0
        times = [max_time * i / 100 for i in range(1, 101)]
        hz = [(self.shape / self.scale) * (t / self.scale) ** (self.shape - 1)
              if (t > 0 and self.scale > 0) else 0.0 for t in times]
        spec.add_trace(times, hz, name="Hazard", trace_type="line", color="", role=ROLE_DATA)
        return spec

    def to_render(self):
        """Primary portrait: the probability plot (hazard alone without a sample)."""
        return self._probability_plot() if self.failure_times else self._hazard()

    def views(self) -> list:
        """Complete portrait: probability plot + survival + hazard (hazard-only
        when the fit carries no sample)."""
        if self.failure_times:
            return [self._probability_plot(), self._survival_curve(), self._hazard()]
        if self.shape and self.scale:
            return [self._hazard()]
        return [self.to_render()]


@dataclass
class ReliabilityResult:
    """Generic reliability analysis result."""

    distribution: str
    parameters: dict[str, float] = field(default_factory=dict)
    mttf: float = 0.0
    b10: float = 0.0  # 10% failure life
    reliability_at: dict[float, float] = field(default_factory=dict)  # {time: R(t)}
    hazard_at: dict[float, float] = field(default_factory=dict)  # {time: h(t)}


def weibull_fit(
    failure_times: list[float] | np.ndarray,
    censored: list[bool] | None = None,
) -> WeibullFit:
    """Fit a 2-parameter Weibull distribution to failure data.

    Args:
        failure_times: Observed failure/censoring times.
        censored: If provided, True = right-censored (survived past this time).
                  MLE with censoring uses simplified approach.

    Returns:
        WeibullFit with shape (β), scale (η), B10 life, failure mode.
    """
    times = np.asarray(failure_times, dtype=float)
    times = times[times > 0]
    n = len(times)

    if n < 3:
        raise ValueError(f"Need at least 3 failure times, got {n}")

    if censored is not None and any(censored):
        # Proper censored Weibull MLE via log-likelihood maximization
        # L = prod(f(t_i) for failures) * prod(R(t_j) for censored)
        cens = np.asarray(censored[:n], dtype=bool)
        fail_times = times[~cens]
        cens_times = times[cens]
        if len(fail_times) < 3:
            raise ValueError("Need at least 3 uncensored failures")

        from scipy.optimize import minimize

        def neg_loglik(params):
            beta, eta = params
            if beta <= 0 or eta <= 0:
                return 1e20
            # Failures: log(f(t)) = log(beta/eta) + (beta-1)*log(t/eta) - (t/eta)^beta
            ll = 0.0
            ll += len(fail_times) * (np.log(beta) - beta * np.log(eta))
            ll += (beta - 1) * np.sum(np.log(fail_times))
            ll -= np.sum((fail_times / eta) ** beta)
            # Censored: log(R(t)) = -(t/eta)^beta
            if len(cens_times) > 0:
                ll -= np.sum((cens_times / eta) ** beta)
            return -ll

        # Initial guess from uncensored-only fit
        s0, _, sc0 = stats.weibull_min.fit(fail_times, floc=0)
        result = minimize(neg_loglik, [s0, sc0], method='Nelder-Mead',
                         options={'xatol': 1e-8, 'fatol': 1e-8, 'maxiter': 5000})
        shape, scale = float(result.x[0]), float(result.x[1])
    else:
        shape, loc, scale = stats.weibull_min.fit(times, floc=0)
        shape = float(shape)
        scale = float(scale)

    shape = float(shape)
    scale = float(scale)

    # B10 life (10th percentile)
    b10 = float(stats.weibull_min.ppf(0.10, shape, 0, scale))

    # MTTF = η * Γ(1 + 1/β)
    mttf = scale * math.gamma(1 + 1 / shape) if shape > 0 else 0.0

    # Median life
    median = float(stats.weibull_min.ppf(0.50, shape, 0, scale))

    # KS test
    ks_stat, ks_p = stats.kstest(times, "weibull_min", args=(shape, 0, scale))

    # Failure mode classification
    if shape < 1:
        mode = "infant_mortality"
    elif abs(shape - 1) < 0.1:
        mode = "random"
    else:
        mode = "wear_out"

    return WeibullFit(
        shape=shape,
        scale=scale,
        location=0.0,
        b10_life=b10,
        mean_life=mttf,
        median_life=median,
        ks_statistic=float(ks_stat),
        ks_p_value=float(ks_p),
        failure_mode=mode,
        failure_times=times.tolist(),  # §5b: views() draw the prob-plot + survival from it
    )


def reliability_function(
    shape: float,
    scale: float,
    times: list[float] | np.ndarray,
    distribution: str = "weibull",
) -> dict[float, float]:
    """Compute R(t) = P(T > t) at given times.

    Args:
        shape: Shape parameter (β for Weibull, σ for lognormal).
        scale: Scale parameter (η for Weibull, median for lognormal).
        times: Time points to evaluate.
        distribution: "weibull", "lognormal", or "exponential".

    Returns:
        Dict of {time: reliability}.
    """
    t_arr = np.asarray(times, dtype=float)
    result = {}

    for t in t_arr:
        if t <= 0:
            result[float(t)] = 1.0
            continue

        if distribution == "weibull":
            r = float(1 - stats.weibull_min.cdf(t, shape, 0, scale))
        elif distribution == "lognormal":
            r = float(1 - stats.lognorm.cdf(t, shape, 0, scale))
        elif distribution == "exponential":
            rate = 1 / scale
            r = float(math.exp(-rate * t))
        else:
            raise ValueError(f"Unknown distribution: {distribution}")

        result[float(t)] = r

    return result


def hazard_function(
    shape: float,
    scale: float,
    times: list[float] | np.ndarray,
    distribution: str = "weibull",
) -> dict[float, float]:
    """Compute hazard rate h(t) = f(t) / R(t) at given times.

    Args:
        shape: Shape parameter.
        scale: Scale parameter.
        times: Time points to evaluate.
        distribution: "weibull", "lognormal", or "exponential".

    Returns:
        Dict of {time: hazard_rate}.
    """
    t_arr = np.asarray(times, dtype=float)
    result = {}

    for t in t_arr:
        if t <= 0:
            result[float(t)] = 0.0
            continue

        if distribution == "weibull":
            # h(t) = (β/η) * (t/η)^(β-1)
            h = (shape / scale) * (t / scale) ** (shape - 1)
        elif distribution == "exponential":
            h = 1 / scale  # constant hazard
        elif distribution == "lognormal":
            f = stats.lognorm.pdf(t, shape, 0, scale)
            r = 1 - stats.lognorm.cdf(t, shape, 0, scale)
            h = f / r if r > 1e-15 else 0.0
        else:
            raise ValueError(f"Unknown distribution: {distribution}")

        result[float(t)] = float(h)

    return result


def exponential_fit(
    failure_times: list[float] | np.ndarray,
) -> ReliabilityResult:
    """Fit exponential distribution (constant hazard rate).

    Args:
        failure_times: Observed failure times.

    Returns:
        ReliabilityResult with rate parameter.
    """
    times = np.asarray(failure_times, dtype=float)
    times = times[times > 0]

    _, scale = stats.expon.fit(times, floc=0)
    rate = 1 / scale
    mttf = scale
    b10 = float(stats.expon.ppf(0.10, 0, scale))

    return ReliabilityResult(
        distribution="exponential",
        parameters={"rate": float(rate), "scale": float(scale)},
        mttf=float(mttf),
        b10=b10,
    )


def lognormal_fit(
    failure_times: list[float] | np.ndarray,
) -> ReliabilityResult:
    """Fit lognormal distribution to failure data.

    Args:
        failure_times: Observed failure times.

    Returns:
        ReliabilityResult with shape (σ) and scale (median) parameters.
    """
    times = np.asarray(failure_times, dtype=float)
    times = times[times > 0]

    shape, loc, scale = stats.lognorm.fit(times, floc=0)
    # shape = σ of log(T), scale = exp(μ) = median
    mu = math.log(scale)
    mttf = math.exp(mu + shape ** 2 / 2)
    b10 = float(stats.lognorm.ppf(0.10, shape, 0, scale))

    return ReliabilityResult(
        distribution="lognormal",
        parameters={"sigma": float(shape), "median": float(scale), "mu": mu},
        mttf=float(mttf),
        b10=b10,
    )
