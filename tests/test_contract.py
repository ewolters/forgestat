"""forgestat as an engine citizen: its result types adopt the forgecore
contract (ResultMixin + Result protocol + views() -> ChartSpec), so the
forgeviz bridge renders them via the contract fallback instead of a bespoke
per-type builder. ACFResult is the first timeseries adopter.
"""

from forgecore import ChartSpec
from forgecore.testing import assert_result_conforms
from forgeviz.renderers import to_svg

from forgestat.timeseries.correlation import ACFResult


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
