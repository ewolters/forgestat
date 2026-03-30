"""Tests for MSA module."""

import numpy as np

from forgestat.msa.gage_rr import crossed_gage_rr
from forgestat.msa.agreement import bland_altman, icc, linearity_bias


class TestCrossedGageRR:
    def test_good_gage(self):
        # 3 parts × 2 operators × 3 reps = 18 measurements
        # Parts differ a lot, gage noise is small
        rng = np.random.default_rng(42)
        measurements = []
        parts = []
        operators = []
        for p_idx, p_mean in enumerate([10, 50, 90]):
            for o_idx in range(2):
                for _ in range(3):
                    measurements.append(p_mean + rng.normal(0, 0.5))
                    parts.append(f"P{p_idx+1}")
                    operators.append(f"Op{o_idx+1}")

        result = crossed_gage_rr(measurements, parts, operators)
        assert result.pct_part > 90  # parts dominate
        assert result.pct_gage_rr < 10  # good gage
        assert result.ndc >= 5  # adequate discrimination

    def test_bad_gage(self):
        # Parts barely differ, gage noise is large
        rng = np.random.default_rng(42)
        measurements = []
        parts = []
        operators = []
        for p_idx, p_mean in enumerate([50, 50.5, 51]):
            for o_idx in range(2):
                for _ in range(3):
                    measurements.append(p_mean + rng.normal(0, 5))
                    parts.append(f"P{p_idx+1}")
                    operators.append(f"Op{o_idx+1}")

        result = crossed_gage_rr(measurements, parts, operators)
        assert result.pct_gage_rr > 50  # gage dominates
        assert result.ndc < 5  # poor discrimination

    def test_anova_table(self):
        rng = np.random.default_rng(42)
        measurements = []
        parts = []
        operators = []
        for p_idx in range(5):
            for o_idx in range(3):
                for _ in range(2):
                    measurements.append(50 + p_idx * 10 + rng.normal(0, 1))
                    parts.append(f"P{p_idx+1}")
                    operators.append(f"Op{o_idx+1}")

        result = crossed_gage_rr(measurements, parts, operators)
        assert len(result.anova_table) == 5  # Part, Op, PxO, Repeat, Total
        assert result.anova_table[0]["source"] == "Part"

    def test_dimensions(self):
        rng = np.random.default_rng(42)
        measurements = []
        parts = []
        operators = []
        for p in range(5):
            for o in range(3):
                for _ in range(2):
                    measurements.append(rng.normal(50, 5))
                    parts.append(f"P{p}")
                    operators.append(f"O{o}")

        result = crossed_gage_rr(measurements, parts, operators)
        assert result.n_parts == 5
        assert result.n_operators == 3
        assert result.n_replicates == 2


class TestICC:
    def test_high_agreement(self):
        # Two raters that agree closely
        rng = np.random.default_rng(42)
        truth = rng.normal(50, 10, 20)
        ratings = np.column_stack([truth + rng.normal(0, 0.5, 20),
                                   truth + rng.normal(0, 0.5, 20)])
        result = icc(ratings)
        assert result.icc > 0.95
        assert result.p_value < 0.001

    def test_low_agreement(self):
        rng = np.random.default_rng(42)
        ratings = rng.normal(50, 10, (20, 3))  # random noise, no agreement
        result = icc(ratings)
        assert result.icc < 0.5

    def test_ci(self):
        rng = np.random.default_rng(42)
        truth = rng.normal(50, 10, 30)
        ratings = np.column_stack([truth + rng.normal(0, 1, 30),
                                   truth + rng.normal(0, 1, 30)])
        result = icc(ratings)
        assert result.ci_lower < result.icc < result.ci_upper


class TestBlandAltman:
    def test_no_bias(self):
        rng = np.random.default_rng(42)
        truth = rng.normal(50, 10, 50)
        m1 = truth + rng.normal(0, 1, 50)
        m2 = truth + rng.normal(0, 1, 50)
        result = bland_altman(m1.tolist(), m2.tolist())
        assert abs(result.mean_diff) < 2  # near zero bias
        assert result.loa_lower < 0 < result.loa_upper

    def test_systematic_bias(self):
        m1 = [10, 20, 30, 40, 50]
        m2 = [12, 22, 32, 42, 52]  # method 2 reads 2 higher
        result = bland_altman(m1, m2)
        assert abs(result.mean_diff - (-2)) < 0.01

    def test_loa_width(self):
        rng = np.random.default_rng(42)
        m1 = rng.normal(50, 5, 100).tolist()
        m2 = (np.array(m1) + rng.normal(0, 2, 100)).tolist()
        result = bland_altman(m1, m2)
        loa_width = result.loa_upper - result.loa_lower
        assert loa_width > 0
        assert loa_width < 20  # reasonable for noise ≈ 2


class TestLinearityBias:
    def test_no_linearity(self):
        # Constant bias of +1, no linearity
        ref = [10, 20, 30, 40, 50]
        meas = [11, 21, 31, 41, 51]
        result = linearity_bias(ref, meas)
        assert abs(result.overall_bias - 1.0) < 0.01
        assert abs(result.linearity_slope) < 0.01  # no linearity

    def test_with_linearity(self):
        # Bias increases with reference value
        ref = [10, 20, 30, 40, 50]
        meas = [10.1, 20.4, 30.9, 41.6, 52.5]  # bias = 0.01 + 0.05*ref
        result = linearity_bias(ref, meas)
        assert result.linearity_slope > 0.01  # positive linearity

    def test_per_level_bias(self):
        ref = [10, 10, 20, 20, 30, 30]
        meas = [11, 11, 21, 21, 31, 31]
        result = linearity_bias(ref, meas)
        assert len(result.bias_per_level) == 3
        assert abs(result.bias_per_level[10.0] - 1.0) < 0.01
