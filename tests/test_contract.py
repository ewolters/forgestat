"""forgestat as an engine citizen: its result types adopt the forgecore
contract (ResultMixin + Result protocol + views() -> ChartSpec), so the
forgeviz bridge renders them via the contract fallback instead of a bespoke
per-type builder. The field-only timeseries family (ACF/CCF/decomposition/
ARIMA/Granger) are adopters.
"""

from forgecore import ChartSpec
from forgecore.testing import assert_result_conforms
from forgeviz.renderers import to_svg

from forgestat.parametric.anova import one_way
from forgestat.parametric.correlation import correlation
from forgestat.core.types import Anova2Result, Anova2Source, ChiSquareResult, ProportionResult
from forgestat.parametric.equivalence import tost
from forgestat.posthoc.comparisons import dunnett, tukey_hsd
from forgestat.parametric.ttest import one_sample, two_sample
from forgestat.nonparametric.rank_tests import (
    kruskal_wallis,
    mann_whitney,
    sign_test,
    wilcoxon_signed_rank,
)
from forgestat.parametric.variance import variance_test
from forgestat.regression.linear import ols
from forgestat.regression.nonlinear import curve_fit
from forgestat.reliability.distributions import WeibullFit, weibull_fit
from forgestat.reliability.survival import kaplan_meier
from forgestat.timeseries.causality import GrangerResult
from forgestat.timeseries.changepoint import pelt
from forgestat.timeseries.correlation import ACFResult, CCFResult
from forgestat.timeseries.decomposition import DecompositionResult
from forgestat.timeseries.forecasting import ARIMAResult, ForecastPoint


def _acf() -> ACFResult:
    return ACFResult(
        acf_values=[1.0, 0.5, 0.2], pacf_values=[1.0, 0.3, 0.1],
        n_lags=2, confidence_bound=0.35,
        significant_acf_lags=[1], significant_pacf_lags=[1],
    )


def test_acf_result_conforms_to_engine_contract():
    assert_result_conforms(_acf())


def test_acf_views_are_acf_and_pacf_correlograms():
    views = _acf().views()
    assert len(views) == 2
    assert all(isinstance(v, ChartSpec) for v in views)
    assert "ACF" in views[0].title
    assert "PACF" in views[1].title


def test_acf_correlogram_carries_confidence_bounds():
    spec = _acf().to_render()
    bounds = sorted(r.value for r in spec.reference_lines)
    assert bounds == [-0.35, 0.35]  # ±95% significance band


def test_acf_views_render_to_svg_through_forgeviz():
    for spec in _acf().views():
        assert "<svg" in to_svg(spec)


def _ccf() -> CCFResult:
    return CCFResult(
        lags=[-2, -1, 0, 1, 2], ccf_values=[0.1, 0.3, 0.6, 0.2, -0.1],
        confidence_bound=0.4, peak_lag=0, peak_value=0.6,
        significant_lags=[0], lead_lag_interpretation="Contemporaneous",
    )


def test_ccf_result_conforms_to_engine_contract():
    assert_result_conforms(_ccf())


def test_ccf_renders_correlogram_with_confidence_bounds():
    spec = _ccf().to_render()
    assert spec.chart_type == "bar"
    bounds = sorted(r.value for r in spec.reference_lines)
    assert bounds == [-0.4, 0.4]


def test_ccf_view_renders_to_svg_through_forgeviz():
    for spec in _ccf().views():
        assert "<svg" in to_svg(spec)


def _decomposition() -> DecompositionResult:
    return DecompositionResult(
        model="additive", period=2,
        observed=[10.0, 12.0, 11.0, 13.0, 12.0, 14.0],
        trend=[10.5, 11.0, 11.5, 12.0, 12.5, 13.0],
        seasonal=[-0.5, 0.5, -0.5, 0.5, -0.5, 0.5],
        residual=[0.0, 0.5, 0.0, 0.5, 0.0, 0.5],
    )


def test_decomposition_result_conforms_to_engine_contract():
    assert_result_conforms(_decomposition())


def test_decomposition_views_are_four_component_panels():
    views = _decomposition().views()
    assert len(views) == 4
    assert all(v.chart_type == "line" for v in views)
    assert [v.title for v in views] == ["Observed", "Trend", "Seasonal", "Residual"]


def test_decomposition_views_render_to_svg_through_forgeviz():
    for spec in _decomposition().views():
        assert "<svg" in to_svg(spec)


def _arima() -> ARIMAResult:
    return ARIMAResult(
        order=(1, 1, 1), aic=42.0, bic=45.0,
        residuals=[0.1, -0.2, 0.0, 0.3, -0.1],
        forecast=[ForecastPoint(1, 11.8, 11.0, 12.6),
                  ForecastPoint(2, 12.0, 11.0, 13.0),
                  ForecastPoint(3, 12.3, 11.1, 13.5)],
    )


def test_arima_result_conforms_to_engine_contract():
    assert_result_conforms(_arima())


def test_arima_views_are_forecast_then_residuals():
    views = _arima().views()
    assert len(views) == 2
    assert len(views[0].traces) == 3  # predicted + lower + upper
    assert views[1].chart_type == "line"


def test_arima_views_render_to_svg_through_forgeviz():
    for spec in _arima().views():
        assert "<svg" in to_svg(spec)


def _granger() -> GrangerResult:
    return GrangerResult(
        x_causes_y=True,
        results_by_lag=[{"lag": 1, "f_stat": 5.0, "p_value": 0.02},
                        {"lag": 2, "f_stat": 3.0, "p_value": 0.08}],
        best_lag=1, best_p_value=0.02, max_lag_tested=2, alpha=0.05,
    )


def test_granger_result_conforms_to_engine_contract():
    assert_result_conforms(_granger())


def test_granger_renders_pvalue_bar_with_alpha_threshold():
    spec = _granger().to_render()
    assert spec.chart_type == "bar"
    assert [r.value for r in spec.reference_lines] == [0.05]


def test_granger_view_renders_to_svg_through_forgeviz():
    for spec in _granger().views():
        assert "<svg" in to_svg(spec)


def _changepoint():
    # A clear level shift at index 25. The default "bic" penalty is too high to
    # flag a step this clean (a known pelt sensitivity); pin a numeric penalty —
    # this exercises the render contract, not detector tuning. Unlike the
    # field-only types, ChangepointResult is data-context: it now carries its
    # own series (§5b data-carrying retrofit) so it self-renders with no data=.
    return pelt([10.0] * 25 + [20.0] * 25, penalty=1.0, min_size=5)


def test_changepoint_result_conforms_to_engine_contract():
    assert_result_conforms(_changepoint())


def test_changepoint_carries_its_own_series():
    assert len(_changepoint().series) == 50  # §5b: the result owns its data


def test_changepoint_to_render_marks_each_changepoint():
    r = _changepoint()
    spec = r.to_render()
    assert spec.chart_type == "line"
    assert len(r.changepoints) >= 1
    assert len(spec.reference_lines) == len(r.changepoints)


def test_changepoint_self_renders_through_bridge_without_data_kwarg():
    # The regression guard: through the REAL bridge, NO data= forwarded —
    # proves the data-carrying retrofit (the old builder needed data=).
    from forgeviz.core.bridge import charts_from_result

    charts = charts_from_result(_changepoint())
    assert len(charts) == 1
    assert charts[0].chart_type == "line"
    assert "<svg" in to_svg(charts[0])


def _regression():
    # Real OLS on y = 2x + noise — the result carries fitted + residuals, which
    # the 4-in-1 residual diagnostics are built from (field-only).
    X = [[float(i)] for i in range(30)]
    y = [2.0 * i + (0.5 if i % 2 else -0.5) for i in range(30)]
    return ols(X, y)


def _nonlinear():
    x = [float(i) for i in range(1, 25)]
    y = [2.0 * (1.0 - 2.718 ** (-0.3 * xi)) for xi in x]
    return curve_fit(x, y, model="exponential")


def test_regression_result_conforms_to_engine_contract():
    assert_result_conforms(_regression())


def test_nonlinear_result_conforms_to_engine_contract():
    assert_result_conforms(_nonlinear())


def test_regression_views_are_the_four_in_one_panel():
    views = _regression().views()
    assert len(views) == 4  # resid-vs-fitted, Q-Q, histogram, resid-vs-order
    assert all(isinstance(v, ChartSpec) for v in views)


def test_regression_self_renders_through_bridge_without_kwargs():
    # The duck-typed regression builder is gone; the result self-renders its
    # 4-in-1 via the contract fallback. No fitted=/residuals= kwargs forwarded.
    from forgeviz.core.bridge import charts_from_result

    charts = charts_from_result(_regression())
    assert len(charts) == 4
    assert all("<svg" in to_svg(c) for c in charts)


def test_nonlinear_self_renders_four_panels_through_bridge():
    from forgeviz.core.bridge import charts_from_result

    charts = charts_from_result(_nonlinear())
    assert len(charts) == 4
    assert all("<svg" in to_svg(c) for c in charts)


def _corr_pair():
    # Two correlated variables — the result is data-context: it now carries the
    # raw columns (§5b) so it self-renders the scatter with no data_dict= kwarg.
    return correlation({"x": [float(i) for i in range(20)],
                        "y": [2.0 * i + (1 if i % 2 else -1) for i in range(20)]})


def _corr_multi():
    rng = [float(i) for i in range(20)]
    return correlation({"a": rng, "b": [2 * v for v in rng],
                        "c": [v * v for v in rng]})


def test_correlation_result_conforms_to_engine_contract():
    assert_result_conforms(_corr_pair())


def test_correlation_carries_its_raw_columns():
    r = _corr_pair()
    assert set(r.data) == {"x", "y"} and len(r.data["x"]) == 20  # §5b


def test_correlation_two_vars_self_renders_one_scatter_via_bridge():
    from forgeviz.core.bridge import charts_from_result

    charts = charts_from_result(_corr_pair())  # NO data_dict=
    assert len(charts) == 1
    assert charts[0].chart_type == "scatter"
    assert "<svg" in to_svg(charts[0])


def test_correlation_three_vars_self_renders_scatter_matrix_via_bridge():
    from forgeviz.core.bridge import charts_from_result

    charts = charts_from_result(_corr_multi())  # NO data_dict=
    assert len(charts) == 9  # 3x3 matrix: diagonal histograms + off-diagonal scatters
    assert all("<svg" in to_svg(c) for c in charts)


def _weibull():
    # Real Weibull fit — the result is data-context: it now carries the raw
    # failure_times (§5b) so the probability plot + survival curve self-render
    # with no failure_times= kwarg.
    return weibull_fit([10.0, 22, 35, 41, 58, 70, 85, 95, 110, 130])


def test_weibull_result_conforms_to_engine_contract():
    assert_result_conforms(_weibull())


def test_weibull_carries_its_failure_times():
    assert len(_weibull().failure_times) == 10  # §5b


def test_weibull_self_renders_full_panel_through_bridge():
    from forgeviz.core.bridge import charts_from_result

    charts = charts_from_result(_weibull())  # NO failure_times= kwarg
    assert len(charts) == 3  # probability plot + survival + hazard
    assert all("<svg" in to_svg(c) for c in charts)


def test_weibull_without_times_falls_back_to_hazard():
    # Parameters alone (no sample) still draw the hazard (bathtub) curve.
    from forgeviz.core.bridge import charts_from_result

    charts = charts_from_result(WeibullFit(shape=2.0, scale=100.0))
    assert len(charts) == 1


def _anova():
    # Three groups with clear mean separation — a real one-way ANOVA. The result
    # carries its raw groups (§5b) so it self-renders a box plot + per-group Q-Q
    # with no groups= kwarg (the old distribution builder needed groups=).
    return one_way(
        [10.0, 11, 9, 10, 12, 8],
        [15.0, 16, 14, 15, 17, 13],
        [20.0, 21, 19, 20, 22, 18],
        labels=["A", "B", "C"],
    )


def test_anova_result_conforms_to_engine_contract():
    assert_result_conforms(_anova())


def test_anova_carries_its_groups():
    r = _anova()
    assert set(r.groups) == {"A", "B", "C"} and len(r.groups["A"]) == 6  # §5b


def test_anova_views_are_boxplot_then_per_group_qq():
    views = _anova().views()
    assert views[0].chart_type == "box_plot"
    assert len(views) == 4  # box plot + one Q-Q per group
    assert all(isinstance(v, ChartSpec) for v in views)


def test_anova_self_renders_through_bridge_without_groups_kwarg():
    from forgeviz.core.bridge import charts_from_result

    charts = charts_from_result(_anova())  # NO groups=
    assert charts[0].chart_type == "box_plot"
    assert len(charts) == 4
    assert all("<svg" in to_svg(c) for c in charts)


def _ttest_two():
    # Two-sample t-test — a group comparison; carries both samples (§5b) and
    # self-renders a box plot + per-group Q-Q with no groups= kwarg.
    return two_sample([10.0, 11, 9, 10, 12, 8], [15.0, 16, 14, 15, 17, 13])


def _ttest_one():
    # One-sample t-test — a single sample; self-renders a histogram + Q-Q.
    return one_sample([10.0, 11, 9, 10, 12, 8, 11, 9, 10, 11], mu=8.0)


def test_ttest_two_sample_conforms_to_engine_contract():
    assert_result_conforms(_ttest_two())


def test_ttest_one_sample_conforms_to_engine_contract():
    assert_result_conforms(_ttest_one())


def test_ttest_carries_its_samples():
    assert len(_ttest_two().samples) == 2  # §5b: both groups
    assert len(_ttest_one().samples) == 1  # §5b: the single sample


def test_ttest_two_sample_views_are_boxplot_then_qq():
    views = _ttest_two().views()
    assert views[0].chart_type == "box_plot"
    assert len(views) == 3  # box plot + one Q-Q per group


def test_ttest_one_sample_views_are_histogram_then_qq():
    views = _ttest_one().views()
    assert views[0].chart_type == "histogram"
    assert len(views) == 2  # histogram + Q-Q


def test_ttest_self_renders_through_bridge_without_kwargs():
    from forgeviz.core.bridge import charts_from_result

    two = charts_from_result(_ttest_two())  # NO groups=
    assert two[0].chart_type == "box_plot" and len(two) == 3
    one = charts_from_result(_ttest_one())  # NO data=
    assert one[0].chart_type == "histogram" and len(one) == 2
    assert all("<svg" in to_svg(c) for c in two + one)


def _mann_whitney():
    # Two-group rank test — carries both samples (§5b), self-renders box + Q-Q.
    return mann_whitney([10.0, 11, 9, 10, 12, 8], [15.0, 16, 14, 15, 17, 13])


def _wilcoxon():
    # Paired signed-rank — self-renders a histogram of the differences.
    return wilcoxon_signed_rank([10.0, 12, 9, 11, 13, 8, 10], [8.0, 9, 7, 10, 11, 6, 9])


def _equivalence():
    # TOST equivalence — two samples, self-renders a box + per-group Q-Q.
    return tost([10.0, 11, 9, 10, 12, 8], [10.2, 10.8, 9.1, 10.3, 11.5, 8.4], margin=2.0)


def test_mann_whitney_conforms_to_engine_contract():
    assert_result_conforms(_mann_whitney())


def test_wilcoxon_conforms_to_engine_contract():
    assert_result_conforms(_wilcoxon())


def test_equivalence_conforms_to_engine_contract():
    assert_result_conforms(_equivalence())


def test_rank_and_equivalence_carry_their_samples():
    assert len(_mann_whitney().samples) == 2  # §5b: both groups
    assert "Differences" in _wilcoxon().samples  # §5b: paired differences
    assert len(_equivalence().samples) == 2  # §5b: both groups


def test_mann_whitney_self_renders_box_through_bridge():
    from forgeviz.core.bridge import charts_from_result

    charts = charts_from_result(_mann_whitney())  # NO groups=
    assert charts[0].chart_type == "box_plot" and len(charts) == 3
    assert all("<svg" in to_svg(c) for c in charts)


def test_wilcoxon_self_renders_histogram_through_bridge():
    from forgeviz.core.bridge import charts_from_result

    charts = charts_from_result(_wilcoxon())  # NO data=
    assert charts[0].chart_type == "histogram"
    assert all("<svg" in to_svg(c) for c in charts)


def test_equivalence_self_renders_box_through_bridge():
    from forgeviz.core.bridge import charts_from_result

    charts = charts_from_result(_equivalence())  # NO groups=
    assert charts[0].chart_type == "box_plot" and len(charts) == 3
    assert all("<svg" in to_svg(c) for c in charts)


def _tukey():
    # Post-hoc across three groups — carries its raw groups (§5b), self-renders
    # a box plot + per-group Q-Q.
    return tukey_hsd(
        [10.0, 11, 9, 10, 12, 8], [15.0, 16, 14, 15, 17, 13],
        [20.0, 21, 19, 20, 22, 18], labels=["A", "B", "C"],
    )


def _dunnett():
    # Control-vs-treatments shape — control + treatments fold into groups too.
    return dunnett(
        [10.0, 11, 9, 10, 12, 8], [15.0, 16, 14, 15, 17, 13],
        [20.0, 21, 19, 20, 22, 18], control_name="Ctrl",
        treatment_names=["T1", "T2"],
    )


def test_tukey_conforms_to_engine_contract():
    assert_result_conforms(_tukey())


def test_dunnett_conforms_to_engine_contract():
    assert_result_conforms(_dunnett())


def test_posthoc_carries_its_groups():
    assert set(_tukey().groups) == {"A", "B", "C"}  # §5b
    assert set(_dunnett().groups) == {"Ctrl", "T1", "T2"}  # control folds in


def test_tukey_self_renders_box_through_bridge():
    from forgeviz.core.bridge import charts_from_result

    charts = charts_from_result(_tukey())  # NO groups=
    assert charts[0].chart_type == "box_plot" and len(charts) == 4  # box + 3 Q-Q
    assert all("<svg" in to_svg(c) for c in charts)


def _chi_square():
    # Field-only: the result already carries observed/expected — no raw arrays.
    return ChiSquareResult(
        test_name="Chi-square", statistic=8.0, p_value=0.018,
        observed=[[10.0, 20.0], [30.0, 15.0]], expected=[[15.0, 15.0], [25.0, 20.0]],
        row_labels=["R1", "R2"], col_labels=["C1", "C2"], cramers_v=0.3,
    )


def _proportion():
    return ProportionResult(
        test_name="2-proportion z-test", statistic=2.1, p_value=0.036,
        p_hat=0.6, p_hat2=0.45, p_diff=0.15, n1=100, n2=120,
    )


def _anova2():
    return Anova2Result(sources=[
        Anova2Source("A", f_statistic=5.0, p_value=0.03, partial_eta_sq=0.20),
        Anova2Source("B", f_statistic=2.0, p_value=0.18, partial_eta_sq=0.10),
        Anova2Source("A:B", f_statistic=1.2, p_value=0.31, partial_eta_sq=0.05),
    ], residual_df=20.0, residual_ss=40.0, residual_ms=2.0)


def test_chi_square_conforms_to_engine_contract():
    assert_result_conforms(_chi_square())


def test_proportion_conforms_to_engine_contract():
    assert_result_conforms(_proportion())


def test_anova2_conforms_to_engine_contract():
    assert_result_conforms(_anova2())


def test_categorical_results_render_their_field_only_views():
    assert _chi_square().to_render().chart_type == "heatmap"
    assert _proportion().to_render().chart_type == "bar"
    assert _anova2().to_render().chart_type == "bar"


def test_categorical_results_self_render_through_bridge():
    from forgeviz.core.bridge import charts_from_result

    for r in (_chi_square(), _proportion(), _anova2()):
        charts = charts_from_result(r)  # NO kwargs
        assert len(charts) == 1 and "<svg" in to_svg(charts[0])


def _kruskal():
    # Bare TestResult from a group test — §5b: carries its raw groups, box + Q-Q.
    return kruskal_wallis([10.0, 11, 9, 10, 12, 8], [15.0, 16, 14, 15, 17, 13],
                          [20.0, 21, 19, 20, 22, 18], labels=["A", "B", "C"])


def _variance():
    return variance_test([10.0, 11, 9, 10, 12, 8], [15.0, 16, 18, 15, 11, 13],
                         labels=["A", "B"])


def _sign():
    # Bare TestResult from a one-sample test — histogram of the sample.
    return sign_test([10.0, 11, 9, 12, 13, 8, 14, 7, 11, 10], median0=9.0)


def test_kruskal_conforms_to_engine_contract():
    assert_result_conforms(_kruskal())


def test_variance_test_conforms_to_engine_contract():
    assert_result_conforms(_variance())


def test_sign_test_conforms_to_engine_contract():
    assert_result_conforms(_sign())


def test_base_testresult_carries_its_samples():
    assert set(_kruskal().samples) == {"A", "B", "C"}  # §5b group test
    assert "Sample" in _sign().samples  # §5b one-sample test


def test_base_testresult_self_renders_through_bridge():
    from forgeviz.core.bridge import charts_from_result

    box = charts_from_result(_kruskal())  # NO groups=
    assert box[0].chart_type == "box_plot" and len(box) == 4
    hist = charts_from_result(_sign())  # NO data=
    assert hist[0].chart_type == "histogram"
    assert all("<svg" in to_svg(c) for c in box + hist)


def _kaplan_meier():
    times = [5.0, 10, 15, 20, 25, 30, 40, 50, 60, 75]
    events = [True, True, False, True, True, False, True, True, False, True]
    return kaplan_meier(times, events)


def test_kaplan_meier_result_conforms_to_engine_contract():
    assert_result_conforms(_kaplan_meier())


def test_kaplan_meier_self_renders_survival_curve_through_bridge():
    # Field-only: KM already carries its computed curve (time, survival), so it
    # self-renders with no failure_times=/censored= kwargs.
    from forgeviz.core.bridge import charts_from_result

    charts = charts_from_result(_kaplan_meier())
    assert len(charts) == 1
    assert "<svg" in to_svg(charts[0])
