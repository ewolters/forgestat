"""Tests for mixed / multilevel models."""

import numpy as np
import pytest

from forgestat.parametric.mixed import (
    MixedResult,
    VarianceDecomposition,
    mixed_model,
    nested_anova,
    variance_decomposition,
)


def _make_nested_data(seed=42):
    """Create nested manufacturing data: measurements within parts within machines.

    3 machines, 4 parts per machine, 3 measurements per part.
    Machine effect: large. Part effect: moderate. Residual: small.
    """
    rng = np.random.RandomState(seed)
    data = []
    machine_effects = {"M1": 0.0, "M2": 5.0, "M3": 10.0}
    for machine, m_eff in machine_effects.items():
        for part_idx in range(4):
            part = f"{machine}_P{part_idx+1}"
            p_eff = rng.normal(0, 2)  # part-level variation
            for _ in range(3):
                measurement = 50 + m_eff + p_eff + rng.normal(0, 0.5)
                data.append({
                    "machine": machine,
                    "part": part,
                    "diameter": measurement,
                })
    return data


def _make_simple_grouped_data(seed=42):
    """Simple random intercept data: 5 groups, 10 obs per group."""
    rng = np.random.RandomState(seed)
    data = []
    group_effects = {"A": 0, "B": 3, "C": -2, "D": 5, "E": 1}
    for group, g_eff in group_effects.items():
        for _ in range(10):
            x = rng.normal(0, 1)
            y = 10 + g_eff + 2 * x + rng.normal(0, 1)
            data.append({"group": group, "x": x, "y": y})
    return data


class TestMixedModel:
    def test_simple_random_intercept(self):
        """Simple random intercept model with one grouping factor."""
        data = _make_simple_grouped_data()
        result = mixed_model(data, response="y", fixed=["x"], random=["group"])
        assert isinstance(result, MixedResult)
        assert result.n_obs == 50
        assert result.n_groups["group"] == 5
        assert result.converged

        # Fixed effect for x should be significant (true coef = 2)
        x_effect = [fe for fe in result.fixed_effects if fe["name"] == "x"][0]
        assert x_effect["significant"], f"x effect p={x_effect['p']}"
        assert abs(x_effect["estimate"] - 2.0) < 1.0

    def test_icc_computation(self):
        """ICC should match manual calculation from variance components."""
        data = _make_simple_grouped_data()
        result = mixed_model(data, response="y", fixed=["x"], random=["group"])

        # Manual ICC: between-group var / total var
        group_var = result.random_effects[0]["variance"]
        total_var = group_var + result.residual_variance
        expected_icc = group_var / total_var if total_var > 0 else 0.0
        assert abs(result.icc - expected_icc) < 1e-10

    def test_icc_is_reasonable(self):
        """ICC should be between 0 and 1 and nonzero when groups differ."""
        data = _make_simple_grouped_data()
        result = mixed_model(data, response="y", fixed=["x"], random=["group"])
        assert 0 < result.icc < 1

    def test_model_fit_statistics(self):
        """AIC, BIC, log-likelihood are computed."""
        data = _make_simple_grouped_data()
        result = mixed_model(data, response="y", fixed=["x"], random=["group"])
        assert result.log_likelihood != 0.0
        assert result.aic > 0
        assert result.bic > 0

    def test_r_squared_marginal_conditional(self):
        """Marginal R2 < Conditional R2 when random effects explain variance."""
        data = _make_simple_grouped_data()
        result = mixed_model(data, response="y", fixed=["x"], random=["group"])
        assert result.r_squared_marginal >= 0
        assert result.r_squared_conditional >= result.r_squared_marginal

    def test_summary_string(self):
        """Summary string is generated and contains key info."""
        data = _make_simple_grouped_data()
        result = mixed_model(data, response="y", fixed=["x"], random=["group"])
        assert "ICC" in result.summary
        assert "Fixed Effects" in result.summary
        assert "Random Effects" in result.summary

    def test_django_style_row_dicts(self):
        """Input is list of dicts, like Django QuerySet.values()."""
        data = [
            {"group": "A", "x": 1.0, "y": 10.0},
            {"group": "A", "x": 2.0, "y": 12.0},
            {"group": "B", "x": 1.0, "y": 15.0},
            {"group": "B", "x": 2.0, "y": 17.0},
            {"group": "C", "x": 1.0, "y": 8.0},
            {"group": "C", "x": 2.0, "y": 10.0},
        ]
        result = mixed_model(data, response="y", fixed=["x"], random=["group"])
        assert result.n_obs == 6
        assert result.n_groups["group"] == 3

    def test_no_fixed_effects(self):
        """Model with only random effects (intercept-only fixed)."""
        data = _make_simple_grouped_data()
        result = mixed_model(data, response="y", fixed=[], random=["group"])
        assert len(result.fixed_effects) == 1  # just intercept
        assert result.fixed_effects[0]["name"] == "Intercept"


class TestVarianceDecomposition:
    def test_sums_to_total(self):
        """Variance components should sum approximately to total variance."""
        data = _make_nested_data()
        decomp = variance_decomposition(data, response="diameter", factors=["machine", "part"])
        assert isinstance(decomp, VarianceDecomposition)
        component_sum = sum(c["variance"] for c in decomp.components)
        # Should be close to total (not exact due to estimation)
        assert abs(component_sum - decomp.total) < decomp.total * 0.5

    def test_percentages_reasonable(self):
        """Percentages should be between 0 and 100."""
        data = _make_nested_data()
        decomp = variance_decomposition(data, response="diameter", factors=["machine"])
        for comp in decomp.components:
            assert 0 <= comp["percent"] <= 100

    def test_machine_explains_most(self):
        """Machine factor should explain most variance in the nested data."""
        data = _make_nested_data()
        decomp = variance_decomposition(data, response="diameter", factors=["machine"])
        machine_comp = [c for c in decomp.components if c["source"] == "machine"][0]
        # Machine effect is large (0, 5, 10), so should explain substantial variance
        assert machine_comp["percent"] > 30

    def test_icc_values(self):
        """ICC values are computed for each factor."""
        data = _make_nested_data()
        decomp = variance_decomposition(data, response="diameter", factors=["machine"])
        assert "machine" in decomp.icc_values
        assert 0 <= decomp.icc_values["machine"] <= 1

    def test_single_factor(self):
        """Works with a single grouping factor."""
        data = [
            {"group": "A", "y": 10}, {"group": "A", "y": 11},
            {"group": "B", "y": 20}, {"group": "B", "y": 21},
        ]
        decomp = variance_decomposition(data, response="y", factors=["group"])
        assert len(decomp.components) == 2  # group + residual


class TestNestedAnova:
    def test_parts_within_machines(self):
        """Nested ANOVA: parts within machines."""
        data = _make_nested_data()
        result = nested_anova(data, response="diameter", factors=["machine", "part"])
        assert isinstance(result, MixedResult)
        assert result.n_obs == 36  # 3 machines * 4 parts * 3 measurements
        assert result.converged

    def test_random_effects_capture_between_group_variance(self):
        """Random effects should have nonzero variance when groups differ."""
        data = _make_nested_data()
        result = nested_anova(data, response="diameter", factors=["machine"])
        machine_re = result.random_effects[0]
        assert machine_re["variance"] > 0

    def test_single_factor(self):
        """Nested ANOVA with a single factor works."""
        data = _make_simple_grouped_data()
        result = nested_anova(data, response="y", factors=["group"])
        assert result.n_groups["group"] == 5

    def test_no_factors_raises(self):
        """At least one factor required."""
        with pytest.raises(ValueError, match="at least one factor"):
            nested_anova([], response="y", factors=[])


class TestEdgeCases:
    def test_single_observation_per_group(self):
        """Works (doesn't crash) with one observation per group."""
        data = [
            {"group": "A", "x": 1.0, "y": 10.0},
            {"group": "B", "x": 2.0, "y": 15.0},
            {"group": "C", "x": 3.0, "y": 20.0},
        ]
        result = mixed_model(data, response="y", fixed=["x"], random=["group"])
        assert result.n_obs == 3
        assert result.n_groups["group"] == 3

    def test_small_data_few_groups(self):
        """Two groups, few observations each."""
        data = [
            {"group": "A", "x": 1.0, "y": 5.0},
            {"group": "A", "x": 2.0, "y": 7.0},
            {"group": "A", "x": 3.0, "y": 9.0},
            {"group": "B", "x": 1.0, "y": 10.0},
            {"group": "B", "x": 2.0, "y": 12.0},
            {"group": "B", "x": 3.0, "y": 14.0},
        ]
        result = mixed_model(data, response="y", fixed=["x"], random=["group"])
        assert result.converged
        # x should be significant (clear linear trend)
        x_effect = [fe for fe in result.fixed_effects if fe["name"] == "x"][0]
        assert x_effect["estimate"] > 0

    def test_fixed_effects_significant_when_means_differ(self):
        """Fixed effects should detect significant group mean differences."""
        rng = np.random.RandomState(42)
        data = []
        # Two treatments with very different means
        for _ in range(30):
            data.append({"treatment": "control", "batch": "B1", "y": rng.normal(10, 1)})
            data.append({"treatment": "drug", "batch": "B1", "y": rng.normal(20, 1)})
        result = mixed_model(data, response="y", fixed=["treatment"], random=["batch"])
        # Treatment effect should be significant
        treatment_effects = [fe for fe in result.fixed_effects if "treatment" in fe["name"]]
        assert len(treatment_effects) > 0
        assert treatment_effects[0]["significant"]
