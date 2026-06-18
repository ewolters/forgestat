"""forgestat as an engine citizen: its result types adopt the forgecore
contract (ResultMixin + Result protocol + views() -> ChartSpec), so the
forgeviz bridge renders them via the contract fallback instead of a bespoke
per-type builder. The field-only timeseries family (ACF/CCF/decomposition/
ARIMA/Granger) are adopters.
"""

from forgecore import ChartSpec
from forgecore.testing import assert_result_conforms
from forgeviz.renderers import to_svg

from forgestat.parametric.correlation import correlation
from forgestat.regression.linear import ols
from forgestat.regression.nonlinear import curve_fit
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
