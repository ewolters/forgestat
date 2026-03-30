"""Calibration adapter for ForgeStat.

Golden reference cases for statistical computations:
- T-tests (known t-statistics, p-values, Cohen's d)
- ANOVA (known F-statistics, SS decomposition, η²)
- Correlation (known r, p-values)
- Chi-square (known χ², Cramér's V)
- Non-parametric (known U, H statistics)
- Post-hoc (known pairwise differences)
"""

from __future__ import annotations



# Golden reference cases — known correct answers
GOLDEN_CASES = [
    # --- T-tests ---
    {
        "case_id": "CAL-STAT-001",
        "description": "One-sample t-test: 10 values, mean≈12, μ₀=0 → significant",
        "test": "ttest_one_sample",
        "input": {
            "data": [10, 12, 11, 13, 14, 12, 11, 10, 15, 13],
            "mu": 0,
        },
        "expected": {"significant": True, "p_value_lt": 0.001, "effect_size_gt": 2.0},
    },
    {
        "case_id": "CAL-STAT-002",
        "description": "Two-sample Welch t-test: groups separated by 10 → p<0.001",
        "test": "ttest_two_sample",
        "input": {
            "x1": [10, 12, 11, 13, 14, 15, 12, 11],
            "x2": [20, 22, 21, 23, 24, 25, 22, 21],
        },
        "expected": {"significant": True, "p_value_lt": 0.001},
    },
    {
        "case_id": "CAL-STAT-003",
        "description": "Paired t-test: consistent +2 shift → significant",
        "test": "ttest_paired",
        "input": {
            "x1": [10, 12, 11, 13, 14, 15, 12, 11, 10, 13],
            "x2": [12, 14, 13, 15, 16, 17, 14, 13, 12, 15],
        },
        "expected": {"significant": True, "mean_diff_lt": 0},
    },
    # --- ANOVA ---
    {
        "case_id": "CAL-STAT-004",
        "description": "One-way ANOVA: 3 groups with 10-unit gaps → F large, η²>0.9",
        "test": "anova_one_way",
        "input": {
            "groups": {
                "Low": [10, 12, 11, 13, 14],
                "Med": [20, 22, 21, 23, 24],
                "High": [30, 32, 31, 33, 34],
            },
        },
        "expected": {"significant": True, "p_value_lt": 0.001, "eta_squared_gt": 0.9},
    },
    {
        "case_id": "CAL-STAT-005",
        "description": "ANOVA SS decomposition: SS_between + SS_within = SS_total",
        "test": "anova_ss_check",
        "input": {
            "groups": {
                "A": [10, 12, 11, 13],
                "B": [20, 22, 21, 23],
            },
        },
        "expected": {"ss_decomposition_valid": True},
    },
    # --- Correlation ---
    {
        "case_id": "CAL-STAT-006",
        "description": "Perfect positive correlation: r=1.0",
        "test": "correlation",
        "input": {
            "data": {"X": [1, 2, 3, 4, 5], "Y": [2, 4, 6, 8, 10]},
        },
        "expected": {"r": 1.0},
    },
    # --- Chi-square ---
    {
        "case_id": "CAL-STAT-007",
        "description": "Chi-square: perfect association in 2×2 → significant",
        "test": "chi_square",
        "input": {
            "observed": [[50, 0], [0, 50]],
        },
        "expected": {"significant": True, "cramers_v_gt": 0.9},
    },
    # --- Nonparametric ---
    {
        "case_id": "CAL-STAT-008",
        "description": "Mann-Whitney U: separated groups → significant",
        "test": "mann_whitney",
        "input": {
            "x1": [10, 12, 11, 13, 14, 15, 12, 11],
            "x2": [20, 22, 21, 23, 24, 25, 22, 21],
        },
        "expected": {"significant": True},
    },
    {
        "case_id": "CAL-STAT-009",
        "description": "Kruskal-Wallis: 3 separated groups → significant",
        "test": "kruskal_wallis",
        "input": {
            "groups": [[10, 12, 11, 13, 14], [20, 22, 21, 23, 24], [30, 32, 31, 33, 34]],
        },
        "expected": {"significant": True},
    },
    # --- Post-hoc ---
    {
        "case_id": "CAL-STAT-010",
        "description": "Tukey HSD: 3 separated groups → all pairs significant",
        "test": "tukey_hsd",
        "input": {
            "groups": [[10, 12, 11, 13, 14], [20, 22, 21, 23, 24], [30, 32, 31, 33, 34]],
            "labels": ["Low", "Med", "High"],
        },
        "expected": {"n_comparisons": 3, "all_significant": True},
    },
    # --- Proportion ---
    {
        "case_id": "CAL-STAT-011",
        "description": "One-proportion z-test: 70/100 vs p₀=0.5 → significant",
        "test": "one_proportion",
        "input": {"successes": 70, "n": 100, "p0": 0.5},
        "expected": {"significant": True, "p_hat": 0.7},
    },
    # --- Equivalence ---
    {
        "case_id": "CAL-STAT-012",
        "description": "TOST: same-distribution samples within margin=2 → equivalent",
        "test": "equivalence",
        "input": {
            "x1": [49.5, 50.2, 49.8, 50.1, 49.9, 50.3, 49.7, 50.0, 49.6, 50.4,
                   49.8, 50.1, 49.9, 50.2, 49.7, 50.3, 49.6, 50.0, 49.8, 50.1],
            "x2": [50.1, 49.8, 50.3, 49.9, 50.0, 49.7, 50.2, 49.6, 50.1, 49.8,
                   50.0, 49.9, 50.2, 49.7, 50.1, 49.8, 50.3, 49.6, 50.0, 49.9],
            "margin": 2.0,
        },
        "expected": {"equivalent": True},
    },
    # --- Phase 2: Regression ---
    {
        "case_id": "CAL-STAT-013",
        "description": "OLS: y = 1 + 2x → R²=1.0, Intercept=1, x=2",
        "test": "ols_regression",
        "input": {
            "X": [[1], [2], [3], [4], [5]],
            "y": [3, 5, 7, 9, 11],
            "feature_names": ["x"],
        },
        "expected": {"r_squared": 1.0, "intercept": 1.0, "x_coef": 2.0},
    },
    {
        "case_id": "CAL-STAT-014",
        "description": "Nonlinear exponential: a≈2, b≈0.5",
        "test": "nonlinear_exp",
        "input": {},
        "expected": {"converged": True, "r_squared_gt": 0.99},
    },
    # --- Phase 2: Power ---
    {
        "case_id": "CAL-STAT-015",
        "description": "Power t-test: d=0.5, n=64 → power≈0.98",
        "test": "power_ttest",
        "input": {"effect_size": 0.5, "n": 64},
        "expected": {"power_gt": 0.90},
    },
    {
        "case_id": "CAL-STAT-016",
        "description": "Sample size for mean CI: σ=5, width=1 → n≈97",
        "test": "sample_size_ci",
        "input": {"target_width": 1.0, "std": 5.0},
        "expected": {"n_gt": 80, "n_lt": 120},
    },
    # --- Phase 2: Quality ---
    {
        "case_id": "CAL-STAT-017",
        "description": "Attribute capability: 50 defects, 10000 units, 5 opp → DPMO=1000",
        "test": "attribute_capability",
        "input": {"defects": 50, "units": 10000, "opportunities": 5},
        "expected": {"dpmo": 1000.0},
    },
    {
        "case_id": "CAL-STAT-018",
        "description": "Variance components: 3 groups with huge gap → ICC>0.9",
        "test": "variance_components",
        "input": {
            "groups": {"A": [10, 11, 10, 11, 10], "B": [50, 51, 50, 51, 50], "C": [90, 91, 90, 91, 90]},
        },
        "expected": {"icc_gt": 0.9},
    },
]


def calibrate():
    """Run all golden reference cases. Standalone entry point."""
    results = []
    for case in GOLDEN_CASES:
        case_id = case["case_id"]
        test = case["test"]
        inp = case["input"]
        exp = case["expected"]

        try:
            actual = _run_case(case_id, test, inp)
            passed = _check_case(actual, exp)
            results.append({"case_id": case_id, "passed": passed, "actual": actual})
        except Exception as e:
            results.append({"case_id": case_id, "passed": False, "error": str(e)})

    passed = sum(1 for r in results if r["passed"])
    return {
        "package": "forgestat",
        "total": len(results),
        "passed": passed,
        "failed": len(results) - passed,
        "results": results,
        "is_calibrated": passed == len(results),
    }


def _run_case(case_id: str, test: str, inp: dict) -> dict:
    """Run a single golden case and return a flat dict with expectation-matching keys."""
    from .parametric.ttest import one_sample, two_sample, paired
    from .parametric.anova import one_way_from_dict
    from .parametric.correlation import correlation
    from .parametric.chi_square import chi_square_independence
    from .parametric.proportion import one_proportion
    from .parametric.equivalence import tost
    from .nonparametric.rank_tests import mann_whitney, kruskal_wallis
    from .posthoc.comparisons import tukey_hsd

    if test == "ttest_one_sample":
        r = one_sample(inp["data"], mu=inp["mu"])
        return {"significant": r.significant, "p_value": r.p_value, "effect_size": r.effect_size}

    elif test == "ttest_two_sample":
        r = two_sample(inp["x1"], inp["x2"])
        return {"significant": r.significant, "p_value": r.p_value}

    elif test == "ttest_paired":
        r = paired(inp["x1"], inp["x2"])
        return {"significant": r.significant, "mean_diff": r.mean_diff}

    elif test == "anova_one_way":
        r = one_way_from_dict(inp["groups"])
        return {"significant": r.significant, "p_value": r.p_value, "eta_squared": r.effect_size}

    elif test == "anova_ss_check":
        r = one_way_from_dict(inp["groups"])
        valid = abs(r.ss_between + r.ss_within - r.ss_total) < 0.01
        return {"ss_decomposition_valid": valid}

    elif test == "correlation":
        r = correlation(inp["data"])
        return {"r": r.pairs[0].r if r.pairs else 0.0}

    elif test == "chi_square":
        r = chi_square_independence(inp["observed"])
        return {"significant": r.significant, "cramers_v": r.cramers_v}

    elif test == "mann_whitney":
        r = mann_whitney(inp["x1"], inp["x2"])
        return {"significant": r.significant}

    elif test == "kruskal_wallis":
        r = kruskal_wallis(*inp["groups"])
        return {"significant": r.significant}

    elif test == "tukey_hsd":
        r = tukey_hsd(*inp["groups"], labels=inp["labels"])
        return {
            "n_comparisons": len(r.comparisons),
            "all_significant": all(c.significant for c in r.comparisons),
        }

    elif test == "one_proportion":
        r = one_proportion(inp["successes"], inp["n"], p0=inp["p0"])
        return {"significant": r.significant, "p_hat": r.p_hat}

    elif test == "equivalence":
        r = tost(inp["x1"], inp["x2"], margin=inp["margin"])
        return {"equivalent": r.equivalent}

    # Phase 2: Regression
    elif test == "ols_regression":
        from .regression.linear import ols as ols_reg
        r = ols_reg(inp["X"], inp["y"], feature_names=inp["feature_names"])
        return {
            "r_squared": r.r_squared,
            "intercept": r.coefficients.get("Intercept", 0),
            "x_coef": r.coefficients.get("x", 0),
        }

    elif test == "nonlinear_exp":
        from .regression.nonlinear import curve_fit as nlin_fit
        import numpy as _np
        x = _np.linspace(0, 5, 50)
        y = 2 * _np.exp(0.5 * x) + _np.random.default_rng(42).normal(0, 0.1, 50)
        r = nlin_fit(x, y, model="exponential")
        return {"converged": r.converged, "r_squared": r.r_squared}

    # Phase 2: Power
    elif test == "power_ttest":
        from .power.sample_size import power_t_test
        r = power_t_test(effect_size=inp["effect_size"], n=inp["n"])
        return {"power": r.power}

    elif test == "sample_size_ci":
        from .power.sample_size import sample_size_for_ci
        n = sample_size_for_ci(target_width=inp["target_width"], std=inp["std"])
        return {"n": n}

    # Phase 2: Quality
    elif test == "attribute_capability":
        from .quality.capability import attribute_capability as attr_cap
        r = attr_cap(inp["defects"], inp["units"], inp["opportunities"])
        return {"dpmo": r.dpmo}

    elif test == "variance_components":
        from .quality.variance_components import one_way_random
        r = one_way_random(inp["groups"])
        return {"icc": r.icc}

    raise ValueError(f"Unknown test: {test}")


def _check_case(actual: dict, expected: dict) -> bool:
    """Check that actual results match expected."""
    for key, val in expected.items():
        if key.endswith("_gt"):
            real_key = key[:-3]
            if actual.get(real_key, 0) <= val:
                return False
        elif key.endswith("_lt"):
            real_key = key[:-3]
            if actual.get(real_key, 0) >= val:
                return False
        else:
            actual_val = actual.get(key)
            if isinstance(val, bool):
                if actual_val != val:
                    return False
            elif isinstance(val, (int, float)):
                if actual_val is None or abs(float(actual_val) - val) > 0.01:
                    return False
    return True


def get_calibration_adapter():
    """ForgeCal adapter protocol."""
    try:
        from forgecal.core import CalibrationAdapter, CalibrationCase, Expectation
    except ImportError:
        return None

    cases = []
    for gc in GOLDEN_CASES:
        expectations = []
        for key, val in gc["expected"].items():
            if key.endswith("_gt"):
                expectations.append(Expectation(
                    key=key[:-3], expected=val, comparison="greater_than",
                ))
            elif key.endswith("_lt"):
                expectations.append(Expectation(
                    key=key[:-3], expected=val, comparison="less_than",
                ))
            elif isinstance(val, bool):
                expectations.append(Expectation(
                    key=key, expected=val, comparison="equals",
                ))
            else:
                expectations.append(Expectation(
                    key=key, expected=val, tolerance=0.01, comparison="abs_within",
                ))

        cases.append(CalibrationCase(
            case_id=gc["case_id"],
            package="forgestat",
            category="statistics",
            analysis_type="stats",
            analysis_id=gc["test"],
            config=gc["input"],
            data={},
            expectations=expectations,
            description=gc["description"],
        ))

    def _run(case):
        gc = next(g for g in GOLDEN_CASES if g["case_id"] == case.case_id)
        return _run_case(case.case_id, gc["test"], gc["input"])

    from forgestat import __version__
    return CalibrationAdapter(package="forgestat", version=__version__, cases=cases, runner=_run)
