"""Result dataclasses for all statistical tests.

Every test function returns a typed result — no dicts.
All result types have .to_dict() for JSON-serializable output.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

from forgecore import ChartSpec, ResultMixin


def _to_dict(obj) -> dict:
    """Recursive dataclass → dict. Handles nested dataclasses and numpy scalars."""
    d = asdict(obj)
    # Convert any numpy scalars that asdict doesn't handle
    def _clean(v):
        if isinstance(v, dict):
            return {k: _clean(val) for k, val in v.items()}
        if isinstance(v, list):
            return [_clean(item) for item in v]
        if hasattr(v, "item"):  # numpy scalar
            return v.item()
        return v
    return _clean(d)


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

    def to_dict(self) -> dict:
        return _to_dict(self)


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

    def to_dict(self) -> dict:
        return _to_dict(self)


@dataclass
class TTestResult(TestResult, ResultMixin):
    """One-sample, two-sample, or paired t-test result."""

    mean1: float = 0.0
    mean2: float | None = None  # None for one-sample
    mean_diff: float = 0.0
    se: float = 0.0
    n1: int = 0
    n2: int | None = None
    samples: dict[str, list[float]] = field(default_factory=dict)  # raw samples — views() draw from it (§5b)

    @property
    def summary(self) -> str:
        return (f"{self.test_name}: t={self.statistic:.3f}, p={self.p_value:.4f}"
                f"{' (significant)' if self.significant else ''}")

    def to_render(self) -> ChartSpec:
        """Primary portrait: box plot (two-sample) or histogram (one-sample)."""
        from ._distribution_views import sample_views
        return sample_views(self.samples)[0]

    def views(self) -> list[ChartSpec]:
        """Full portrait from the raw samples, shaped by sample count."""
        from ._distribution_views import sample_views
        return sample_views(self.samples)


@dataclass
class AnovaResult(TestResult, ResultMixin):
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
    groups: dict[str, list[float]] = field(default_factory=dict)  # raw samples — views() draw from it (§5b)

    @property
    def summary(self) -> str:
        return (f"One-way ANOVA, {len(self.group_means)} groups; "
                f"F={self.statistic:.3f}, p={self.p_value:.4f}"
                f"{' (significant)' if self.significant else ''}")

    def to_render(self) -> ChartSpec:
        """Primary portrait: a box plot comparing the groups."""
        from ._distribution_views import box_views
        return box_views(self.groups, "Group Comparison")[0]

    def views(self) -> list[ChartSpec]:
        """Box plot + a normal Q-Q per group, all from the raw samples."""
        from ._distribution_views import box_views
        return box_views(self.groups, "Group Comparison")


@dataclass
class Anova2Result:
    """Two-way ANOVA result."""

    sources: list[Anova2Source] = field(default_factory=list)
    residual_df: float = 0.0
    residual_ss: float = 0.0
    residual_ms: float = 0.0
    assumptions: list[AssumptionCheck] = field(default_factory=list)

    def to_dict(self) -> dict:
        return _to_dict(self)


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
class CorrelationResult(ResultMixin):
    """Correlation analysis result."""

    method: str  # "pearson", "spearman", "kendall"
    pairs: list[CorrelationPair] = field(default_factory=list)
    matrix: dict[str, dict[str, float]] = field(default_factory=dict)
    assumptions: list[AssumptionCheck] = field(default_factory=list)
    data: dict[str, list[float]] = field(default_factory=dict)  # raw columns — views() draw from it (§5b)

    def to_dict(self) -> dict:
        return _to_dict(self)

    @property
    def summary(self) -> str:
        if self.pairs:
            top = self.pairs[0]
            return (f"{self.method} correlation, {len(self.pairs)} pair(s); "
                    f"strongest {top.var1}~{top.var2} r={top.r:.3f}")
        return f"{self.method} correlation"

    def _scatter(self, xname: str, yname: str, title: str = ""):
        """Scatter of two columns + a least-squares trend line, theme-neutral."""
        import numpy as np
        from forgecore import ROLE_CONTROL_LIMIT, ROLE_DATA, ChartSpec

        x = list(self.data.get(xname, []))
        y = list(self.data.get(yname, []))
        spec = ChartSpec(title=title or f"{xname} vs {yname}", chart_type="scatter",
                         x_axis={"label": xname}, y_axis={"label": yname})
        spec.add_trace(x, y, trace_type="scatter", color="", role=ROLE_DATA)
        if len(set(x)) >= 2:
            slope, intercept = np.polyfit(x, y, 1)
            ends = [min(x), max(x)]
            spec.add_trace(ends, [slope * v + intercept for v in ends],
                           trace_type="line", dash="dashed", color="", role=ROLE_CONTROL_LIMIT)
        return spec

    def _hist(self, name: str):
        import numpy as np
        from forgecore import ROLE_DATA, ChartSpec

        counts, edges = np.histogram(list(self.data.get(name, [])), bins=10)
        centers = [f"{(edges[i] + edges[i + 1]) / 2:.2g}" for i in range(len(counts))]
        spec = ChartSpec(title=name, chart_type="bar",
                         x_axis={"label": name}, y_axis={"label": "Frequency"})
        spec.add_trace(centers, counts.tolist(), trace_type="bar", color="", role=ROLE_DATA)
        return spec

    def _cols(self) -> list[str]:
        return [c for c in self.data if self.data[c]]

    def to_render(self):
        """Primary portrait: scatter of the strongest correlated pair."""
        cols = self._cols()
        if self.pairs and self.data:
            return self._scatter(self.pairs[0].var1, self.pairs[0].var2)
        if len(cols) >= 2:
            return self._scatter(cols[0], cols[1])
        from forgecore import ChartSpec
        return ChartSpec(title="Correlation", chart_type="scatter")

    def views(self) -> list:
        """Complete portrait: one scatter (2 vars) or a scatter matrix (>2)."""
        cols = self._cols()
        if len(cols) > 2:
            return [self._hist(xi) if xi == yi else self._scatter(xi, yi)
                    for yi in cols for xi in cols]
        if len(cols) == 2:
            return [self._scatter(cols[0], cols[1])]
        return [self.to_render()]


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
class EquivalenceResult(ResultMixin):
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
    samples: dict[str, list[float]] = field(default_factory=dict)  # raw samples — views() draw from it (§5b)

    def to_dict(self) -> dict:
        return _to_dict(self)

    @property
    def summary(self) -> str:
        return (f"TOST: mean diff={self.mean_diff:.3f}, margin=±{self.margin:.3f}; "
                f"{'equivalent' if self.equivalent else 'not equivalent'} "
                f"(p={self.p_tost:.4f})")

    def to_render(self) -> ChartSpec:
        from ._distribution_views import sample_views
        return sample_views(self.samples, "Group Comparison")[0]

    def views(self) -> list[ChartSpec]:
        from ._distribution_views import sample_views
        return sample_views(self.samples, "Group Comparison")


@dataclass
class RankTestResult(TestResult, ResultMixin):
    """Non-parametric rank test result."""

    median1: float | None = None
    median2: float | None = None
    median_diff: float | None = None
    n1: int = 0
    n2: int | None = None
    samples: dict[str, list[float]] = field(default_factory=dict)  # raw samples — views() draw from it (§5b)

    @property
    def summary(self) -> str:
        return (f"{self.test_name}: stat={self.statistic:.3f}, p={self.p_value:.4f}"
                f"{' (significant)' if self.significant else ''}")

    def to_render(self) -> ChartSpec:
        from ._distribution_views import sample_views
        return sample_views(self.samples)[0]

    def views(self) -> list[ChartSpec]:
        from ._distribution_views import sample_views
        return sample_views(self.samples)


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
class PostHocResult(ResultMixin):
    """Result of a post-hoc multiple comparison procedure."""

    test_name: str  # "tukey_hsd", "dunnett", "games_howell", "dunn"
    comparisons: list[PostHocComparison] = field(default_factory=list)
    alpha: float = 0.05
    correction: str = ""  # "studentized_range", "bonferroni", etc.
    group_means: dict[str, float] = field(default_factory=dict)
    control_group: str | None = None  # for Dunnett
    groups: dict[str, list[float]] = field(default_factory=dict)  # raw samples — views() draw from it (§5b)

    def to_dict(self) -> dict:
        return _to_dict(self)

    @property
    def summary(self) -> str:
        n_sig = sum(1 for c in self.comparisons if c.significant)
        return (f"{self.test_name}: {len(self.comparisons)} comparisons, "
                f"{n_sig} significant")

    def to_render(self) -> ChartSpec:
        from ._distribution_views import box_views
        return box_views(self.groups, "Group Comparison")[0]

    def views(self) -> list[ChartSpec]:
        from ._distribution_views import box_views
        return box_views(self.groups, "Group Comparison")
