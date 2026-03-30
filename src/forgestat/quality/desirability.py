"""Multi-response desirability optimization (Derringer-Suich).

Converts multiple response objectives into a single composite score.
Used in DOE optimization: find settings that simultaneously satisfy all responses.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field


@dataclass
class IndividualDesirability:
    """Desirability for one response."""

    name: str
    value: float
    desirability: float  # 0-1
    target: float
    lower: float | None = None
    upper: float | None = None
    weight: float = 1.0
    goal: str = "target"  # "target", "maximize", "minimize"


@dataclass
class DesirabilityResult:
    """Composite desirability result."""

    composite_d: float = 0.0  # geometric mean of individual desirabilities
    individual: list[IndividualDesirability] = field(default_factory=list)
    all_satisfied: bool = False


def derringer_suich(responses: list[dict]) -> DesirabilityResult:
    """Derringer-Suich desirability function.

    Each response defines a goal (target, maximize, or minimize) with bounds.
    Individual desirabilities are combined via weighted geometric mean.

    Args:
        responses: List of response dicts, each with:
            - name: Response name
            - value: Current/predicted value
            - goal: "target", "maximize", or "minimize"
            - target: Target value (for "target" goal)
            - lower: Lower bound (value below this → d=0)
            - upper: Upper bound (value above this → d=0)
            - weight: Shape weight (1=linear, >1=convex, <1=concave)
            - importance: Relative importance (for composite weighting)

    Returns:
        DesirabilityResult with individual and composite desirabilities.
    """
    individual = []
    importances = []

    for resp in responses:
        name = resp["name"]
        val = resp["value"]
        goal = resp.get("goal", "target")
        target = resp.get("target", val)
        lower = resp.get("lower")
        upper = resp.get("upper")
        weight = resp.get("weight", 1.0)
        importance = resp.get("importance", 1.0)

        if goal == "target":
            d = _target_desirability(val, lower, target, upper, weight)
        elif goal == "maximize":
            d = _larger_is_better(val, lower, upper or target, weight)
        elif goal == "minimize":
            d = _smaller_is_better(val, lower or target, upper, weight)
        else:
            d = 0.0

        individual.append(IndividualDesirability(
            name=name, value=val, desirability=d,
            target=target, lower=lower, upper=upper,
            weight=weight, goal=goal,
        ))
        importances.append(importance)

    # Composite: weighted geometric mean
    if individual and all(d.desirability > 0 for d in individual):
        total_imp = sum(importances)
        log_d = sum(
            imp * math.log(d.desirability)
            for d, imp in zip(individual, importances)
        )
        composite = math.exp(log_d / total_imp) if total_imp > 0 else 0
    else:
        composite = 0.0

    return DesirabilityResult(
        composite_d=composite,
        individual=individual,
        all_satisfied=all(d.desirability > 0 for d in individual),
    )


def _target_desirability(value, lower, target, upper, weight):
    """Two-sided: maximize at target, zero at bounds."""
    if lower is None or upper is None:
        return 1.0 if abs(value - target) < 1e-10 else 0.0

    if value < lower or value > upper:
        return 0.0
    if abs(value - target) < 1e-10:
        return 1.0
    if value < target:
        return ((value - lower) / (target - lower)) ** weight if target > lower else 0.0
    else:
        return ((upper - value) / (upper - target)) ** weight if upper > target else 0.0


def _larger_is_better(value, lower, upper, weight):
    """One-sided: larger values better."""
    if lower is None:
        lower = value - 1
    if value <= lower:
        return 0.0
    if value >= upper:
        return 1.0
    return ((value - lower) / (upper - lower)) ** weight


def _smaller_is_better(value, lower, upper, weight):
    """One-sided: smaller values better."""
    if upper is None:
        upper = value + 1
    if value >= upper:
        return 0.0
    if value <= lower:
        return 1.0
    return ((upper - value) / (upper - lower)) ** weight
