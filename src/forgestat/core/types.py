"""Result dataclasses for all statistical tests.

Every test function returns a typed result — no dicts.
SVEND's thin layer converts these to its presentation format.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class AssumptionCheck:
    """Result of checking a statistical assumption."""

    name: str  # "normality", "equal_variance", "outliers"
    test_name: str  # "Shapiro-Wilk", "Levene", "IQR"
    statistic: float | None = None
    p_value: float | None = None
    passed: bool = True
    detail: str = ""
    suggestion: str = ""


@dataclass
class TestResult:
    """Result of a hypothesis test."""

    test_name: str
    statistic: float
    p_value: float
    df: float | None = None
    ci_lower: float | None = None
    ci_upper: float | None = None
    effect_size: float | None = None
    effect_size_type: str = ""  # "cohens_d", "eta_squared", "cramers_v", "r"
    effect_label: str = ""  # "negligible", "small", "medium", "large"
    alpha: float = 0.05
    significant: bool = False
    method: str = ""  # "welch", "pooled", etc.
    assumptions: list[AssumptionCheck] = field(default_factory=list)
    extra: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        self.significant = bool(self.p_value < self.alpha)


@dataclass
class TTestResult(TestResult):
    """One-sample, two-sample, or paired t-test result."""

    mean1: float = 0.0
    mean2: float | None = None  # None for one-sample
    mean_diff: float = 0.0
    se: float = 0.0
    n1: int = 0
    n2: int | None = None


@dataclass
class AnovaResult(TestResult):
    """ANOVA F-test result."""

    df_between: float = 0.0
    df_within: float = 0.0
    ss_between: float = 0.0
    ss_within: float = 0.0
    ss_total: float = 0.0
    ms_between: float = 0.0
    ms_within: float = 0.0
    omega_squared: float | None = None
    group_means: dict[str, float] = field(default_factory=dict)
    group_ns: dict[str, int] = field(default_factory=dict)


@dataclass
class Anova2Result:
    """Two-way ANOVA result."""

    sources: list[Anova2Source] = field(default_factory=list)
    residual_df: float = 0.0
    residual_ss: float = 0.0
    residual_ms: float = 0.0
    assumptions: list[AssumptionCheck] = field(default_factory=list)


@dataclass
class Anova2Source:
    """One source row in a two-way ANOVA table."""

    source: str  # "A", "B", "A:B"
    ss: float = 0.0
    df: float = 0.0
    ms: float = 0.0
    f_statistic: float = 0.0
    p_value: float = 0.0
    partial_eta_sq: float = 0.0


@dataclass
class CorrelationResult:
    """Correlation analysis result."""

    method: str  # "pearson", "spearman", "kendall"
    pairs: list[CorrelationPair] = field(default_factory=list)
    matrix: dict[str, dict[str, float]] = field(default_factory=dict)
    assumptions: list[AssumptionCheck] = field(default_factory=list)


@dataclass
class CorrelationPair:
    """One pairwise correlation."""

    var1: str
    var2: str
    r: float
    p_value: float
    n: int
    ci_lower: float | None = None
    ci_upper: float | None = None
    r_squared: float = 0.0


@dataclass
class ChiSquareResult(TestResult):
    """Chi-square test result."""

    observed: list[list[float]] = field(default_factory=list)
    expected: list[list[float]] = field(default_factory=list)
    row_labels: list[str] = field(default_factory=list)
    col_labels: list[str] = field(default_factory=list)
    cramers_v: float = 0.0


@dataclass
class ProportionResult(TestResult):
    """Proportion test result (1-sample or 2-sample)."""

    p_hat: float = 0.0
    p_hat2: float | None = None
    p_diff: float | None = None
    n1: int = 0
    n2: int | None = None


@dataclass
class EquivalenceResult:
    """TOST equivalence test result."""

    mean_diff: float = 0.0
    margin: float = 0.0
    t_lower: float = 0.0
    t_upper: float = 0.0
    p_lower: float = 0.0
    p_upper: float = 0.0
    p_tost: float = 0.0
    equivalent: bool = False
    ci_lower: float = 0.0
    ci_upper: float = 0.0
    ci_level: float = 0.90
    effect_size: float | None = None
    assumptions: list[AssumptionCheck] = field(default_factory=list)


@dataclass
class RankTestResult(TestResult):
    """Non-parametric rank test result."""

    median1: float | None = None
    median2: float | None = None
    median_diff: float | None = None
    n1: int = 0
    n2: int | None = None


@dataclass
class PostHocComparison:
    """One pairwise comparison from a post-hoc test."""

    group1: str
    group2: str
    mean_diff: float
    se: float = 0.0
    t_or_q: float = 0.0
    p_value: float = 0.0
    ci_lower: float | None = None
    ci_upper: float | None = None
    significant: bool = False
    reject: bool = False


@dataclass
class PostHocResult:
    """Result of a post-hoc multiple comparison procedure."""

    test_name: str  # "tukey_hsd", "dunnett", "games_howell", "dunn"
    comparisons: list[PostHocComparison] = field(default_factory=list)
    alpha: float = 0.05
    correction: str = ""  # "studentized_range", "bonferroni", etc.
    group_means: dict[str, float] = field(default_factory=dict)
    control_group: str | None = None  # for Dunnett
