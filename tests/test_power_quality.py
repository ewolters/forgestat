"""Tests for power analysis and quality statistics."""

import numpy as np

from forgestat.power.sample_size import (
    power_anova,
    power_chi_square,
    power_proportion,
    power_t_test,
    sample_size_for_ci,
)
from forgestat.quality.acceptance import attribute_plan, variable_plan
from forgestat.quality.capability import attribute_capability, nonnormal_capability
from forgestat.quality.variance_components import one_way_random


class TestPowerTTest:
    def test_power_increases_with_n(self):
        r1 = power_t_test(effect_size=0.5, n=10)
        r2 = power_t_test(effect_size=0.5, n=50)
        assert r2.power > r1.power

    def test_sample_size_for_target_power(self):
        result = power_t_test(effect_size=0.5, power=0.80)
        assert result.sample_size > 0
        assert result.power >= 0.80

    def test_two_sample(self):
        result = power_t_test(effect_size=0.8, n=25, test_type="two_sample")
        assert 0 < result.power < 1

    def test_large_effect_high_power(self):
        result = power_t_test(effect_size=1.5, n=30)
        assert result.power > 0.95


class TestPowerAnova:
    def test_power_with_groups(self):
        result = power_anova(effect_size=0.4, k=3, n_per_group=30)
        assert result.power > 0.5

    def test_sample_size(self):
        result = power_anova(effect_size=0.25, k=4, power=0.80)
        assert result.sample_size > 10


class TestPowerProportion:
    def test_one_proportion(self):
        result = power_proportion(p1=0.6, p0=0.5, n=100)
        assert 0 < result.power < 1

    def test_two_proportions_sample_size(self):
        result = power_proportion(p1=0.6, p2=0.4, power=0.80)
        assert result.sample_size > 0
        assert result.power >= 0.80


class TestPowerChiSquare:
    def test_basic(self):
        result = power_chi_square(effect_size=0.3, df=4, n=100)
        assert 0 < result.power < 1

    def test_sample_size(self):
        result = power_chi_square(effect_size=0.3, df=2, power=0.80)
        assert result.sample_size > 0


class TestSampleSizeCI:
    def test_mean_ci(self):
        n = sample_size_for_ci(target_width=1.0, std=5.0)
        assert n > 50  # should need a decent sample for narrow CI

    def test_proportion_ci(self):
        n = sample_size_for_ci(target_width=0.05, proportion=0.5)
        assert n > 300  # worst case (p=0.5) needs ~385


class TestVarianceComponents:
    def test_between_dominates(self):
        groups = {
            "A": [10, 11, 10, 11, 10],
            "B": [50, 51, 50, 51, 50],
            "C": [90, 91, 90, 91, 90],
        }
        result = one_way_random(groups)
        assert result.icc > 0.9  # between-group variance dominates
        assert result.components[0].pct_contribution > 90

    def test_within_dominates(self):
        rng = np.random.default_rng(42)
        groups = {
            "A": rng.normal(50, 10, 20).tolist(),
            "B": rng.normal(50, 10, 20).tolist(),
        }
        result = one_way_random(groups)
        assert result.icc < 0.3  # groups have same mean, within dominates


class TestAcceptanceSampling:
    def test_attribute_plan(self):
        plan = attribute_plan(aql=0.01, ltpd=0.05, producer_risk=0.05, consumer_risk=0.10)
        assert plan.plan_type == "attribute"
        assert plan.sample_size > 0
        assert plan.acceptance_number >= 0
        assert len(plan.oc_curve) > 0

    def test_variable_plan(self):
        plan = variable_plan(aql=0.01, ltpd=0.05)
        assert plan.plan_type == "variable"
        assert plan.sample_size > 0
        assert plan.k_value > 0
        assert len(plan.oc_curve) > 0

    def test_tighter_plan_needs_more_samples(self):
        loose = attribute_plan(aql=0.05, ltpd=0.15)
        tight = attribute_plan(aql=0.01, ltpd=0.05)
        assert tight.sample_size >= loose.sample_size


class TestAttributeCapability:
    def test_basic(self):
        result = attribute_capability(defects=50, units=10000, opportunities=5)
        assert result.dpmo == 1000  # 50 / (10000*5) * 1e6
        assert result.yield_pct > 99
        assert result.sigma_short_term > 4  # good process

    def test_zero_defects(self):
        result = attribute_capability(defects=0, units=1000)
        assert result.dpmo == 0
        assert result.yield_pct == 100


class TestNonNormalCapability:
    def test_normal_data(self):
        rng = np.random.default_rng(42)
        data = rng.normal(50, 2, 200).tolist()
        result = nonnormal_capability(data, lsl=40, usl=60)
        assert result.cnp > 1.0  # ±5σ spec on 2σ process
        assert result.cnpk > 0.5

    def test_skewed_data(self):
        rng = np.random.default_rng(42)
        data = rng.lognormal(3, 0.5, 200).tolist()
        result = nonnormal_capability(data, lsl=0, usl=100)
        assert result.median > 0
        assert result.p_low < result.median < result.p_high
