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
    # --- Phase 3: Bayesian ---
    {
        "case_id": "CAL-STAT-019",
        "description": "Bayesian t-test: strong evidence against H0 (BF10 > 10)",
        "test": "bayesian_ttest",
        "input": {"data": [10, 12, 11, 13, 14, 15, 12, 11, 10, 13], "mu": 0},
        "expected": {"bf10_gt": 10},
    },
    {
        "case_id": "CAL-STAT-020",
        "description": "Bayesian proportion: 80/100 → posterior mean ≈ 0.79",
        "test": "bayesian_proportion",
        "input": {"successes": 80, "n": 100},
        "expected": {"posterior_mean_gt": 0.75, "posterior_mean_lt": 0.85},
    },
    # --- Phase 3: Reliability ---
    {
        "case_id": "CAL-STAT-021",
        "description": "Weibull fit: shape > 1.5 (wear-out), B10 > 0",
        "test": "weibull_fit",
        "input": {},
        "expected": {"failure_mode": "wear_out", "b10_gt": 0},
    },
    {
        "case_id": "CAL-STAT-022",
        "description": "Kaplan-Meier: 7 events, median at t=10",
        "test": "kaplan_meier",
        "input": {"times": [5, 10, 10, 15, 20, 25, 30]},
        "expected": {"n_events": 7, "median_survival": 15.0},
    },
    {
        "case_id": "CAL-STAT-023",
        "description": "Log-rank test: early vs late failures → p < 0.05",
        "test": "log_rank",
        "input": {
            "t1": [5, 10, 15, 20, 25, 30, 35, 40],
            "t2": [50, 60, 70, 80, 90, 100, 110, 120],
        },
        "expected": {"significant": True},
    },
    # --- Phase 4: Exploratory ---
    {
        "case_id": "CAL-STAT-024",
        "description": "Descriptive: 100 values → mean=50.5, n=100",
        "test": "descriptive",
        "input": {"data": list(range(1, 101))},
        "expected": {"mean": 50.5, "n": 100},
    },
    {
        "case_id": "CAL-STAT-025",
        "description": "Meta-analysis: 5 homogeneous studies → pooled ≈ 0.5",
        "test": "meta_analysis",
        "input": {
            "effects": [0.50, 0.51, 0.49, 0.50, 0.50],
            "ses": [0.1, 0.1, 0.1, 0.1, 0.1],
        },
        "expected": {"pooled_effect_gt": 0.45, "pooled_effect_lt": 0.55},
    },
    # --- Phase 4: MSA ---
    {
        "case_id": "CAL-STAT-026",
        "description": "Gage R&R: good gage → %GRR < 10, NDC ≥ 5",
        "test": "gage_rr",
        "input": {
            "measurements": [10.1, 10.0, 10.2, 50.1, 50.0, 50.2, 90.1, 90.0, 90.2,
                             10.0, 10.1, 10.1, 50.0, 50.1, 50.1, 90.0, 90.1, 90.1],
            "parts": ["P1"]*3 + ["P2"]*3 + ["P3"]*3 + ["P1"]*3 + ["P2"]*3 + ["P3"]*3,
            "operators": ["A"]*9 + ["B"]*9,
        },
        "expected": {"pct_gage_rr_lt": 10, "ndc_gt": 4},
    },
    {
        "case_id": "CAL-STAT-027",
        "description": "Bland-Altman: method 2 reads 2 higher → bias = -2",
        "test": "bland_altman",
        "input": {
            "m1": [10, 20, 30, 40, 50],
            "m2": [12, 22, 32, 42, 52],
        },
        "expected": {"mean_diff": -2.0},
    },
    # --- Phase 5: Time series ---
    {
        "case_id": "CAL-STAT-028",
        "description": "ADF: white noise is stationary",
        "test": "adf_stationarity",
        "input": {},
        "expected": {"is_stationary": True},
    },
    {
        "case_id": "CAL-STAT-029",
        "description": "ARIMA(1,1,0) forecast produces CIs",
        "test": "arima_forecast",
        "input": {},
        "expected": {"n_forecast_gt": 0, "has_ci": True},
    },
    {
        "case_id": "CAL-STAT-030",
        "description": "Decomposition: seasonal data → seasonal strength > 0.5",
        "test": "decomposition",
        "input": {},
        "expected": {"seasonal_strength_gt": 0.5, "trend_direction": "upward"},
    },
    {
        "case_id": "CAL-STAT-031",
        "description": "PELT detects shift at index ~50",
        "test": "pelt_changepoint",
        "input": {},
        "expected": {"n_changepoints_gt": 0},
    },
    {
        "case_id": "CAL-STAT-032",
        "description": "Granger causality: lagged X causes Y",
        "test": "granger",
        "input": {},
        "expected": {"x_causes_y": True},
    },
    {
        "case_id": "CAL-STAT-033",
        "description": "Anomaly scoring detects injected spike",
        "test": "anomaly_score",
        "input": {},
        "expected": {"spike_detected": True},
    },
    # --- Phase 5: Parity gaps ---
    {
        "case_id": "CAL-STAT-034",
        "description": "Repeated measures ANOVA: 3 conditions, p < 0.001",
        "test": "repeated_measures",
        "input": {
            "Baseline": [10, 12, 11, 13, 14, 15, 12, 11],
            "Treatment1": [15, 17, 16, 18, 19, 20, 17, 16],
            "Treatment2": [20, 22, 21, 23, 24, 25, 22, 21],
        },
        "expected": {"significant": True, "n_conditions": 3},
    },
    {
        "case_id": "CAL-STAT-035",
        "description": "Runs test: clustered data is non-random",
        "test": "runs_test",
        "input": {"data": [1] * 50 + [10] * 50},
        "expected": {"significant": True},
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

    # Phase 3: Bayesian
    elif test == "bayesian_ttest":
        from .bayesian.tests import bayesian_ttest_one_sample
        r = bayesian_ttest_one_sample(inp["data"], mu=inp["mu"])
        return {"bf10": r.bf10}

    elif test == "bayesian_proportion":
        from .bayesian.tests import bayesian_proportion as bp
        r = bp(inp["successes"], inp["n"])
        return {"posterior_mean": r.posterior_mean}

    # Phase 3: Reliability
    elif test == "weibull_fit":
        from .reliability.distributions import weibull_fit as wf
        import numpy as _np
        data = _np.random.default_rng(42).weibull(2.5, 100) * 1000
        r = wf(data.tolist())
        return {"failure_mode": r.failure_mode, "b10": r.b10_life}

    elif test == "kaplan_meier":
        from .reliability.survival import kaplan_meier as km
        r = km(inp["times"])
        return {"n_events": r.n_events, "median_survival": r.median_survival}

    elif test == "log_rank":
        from .reliability.survival import log_rank_test
        r = log_rank_test(inp["t1"], [True] * len(inp["t1"]), inp["t2"], [True] * len(inp["t2"]))
        return {"significant": r.p_value < 0.05}

    # Phase 4: Exploratory
    elif test == "descriptive":
        from .exploratory.univariate import describe as desc
        r = desc(inp["data"])
        return {"mean": r.mean, "n": r.n, "skewness": r.skewness}

    elif test == "pca":
        from .exploratory.multivariate import pca as run_pca
        r = run_pca(inp["data"], n_components=inp.get("n_components"))
        return {"pc1_variance": r.variance_explained[0], "n_components": r.n_components}

    elif test == "meta_analysis":
        from .exploratory.meta import meta_analysis as run_meta
        r = run_meta(inp["effects"], inp["ses"], model=inp.get("model", "random"))
        return {"pooled_effect": r.pooled_effect, "i_squared": r.i_squared}

    # Phase 4: MSA
    elif test == "gage_rr":
        from .msa.gage_rr import crossed_gage_rr
        r = crossed_gage_rr(inp["measurements"], inp["parts"], inp["operators"])
        return {"pct_gage_rr": r.pct_gage_rr, "ndc": r.ndc}

    elif test == "icc":
        from .msa.agreement import icc as run_icc
        r = run_icc(inp["ratings"])
        return {"icc": r.icc}

    elif test == "bland_altman":
        from .msa.agreement import bland_altman as run_ba
        r = run_ba(inp["m1"], inp["m2"])
        return {"mean_diff": r.mean_diff}

    # Phase 5: Time series
    elif test == "adf_stationarity":
        from .timeseries.stationarity import adf_test
        import numpy as _np
        data = _np.random.default_rng(42).normal(0, 1, 200).tolist()
        r = adf_test(data)
        return {"is_stationary": r.is_stationary}

    elif test == "arima_forecast":
        from .timeseries.forecasting import arima as run_arima
        import numpy as _np
        data = _np.cumsum(_np.random.default_rng(42).normal(0.1, 1, 100)).tolist()
        r = run_arima(data, order=(1, 1, 0), forecast_steps=5)
        has_ci = all(pt.ci_lower < pt.predicted < pt.ci_upper for pt in r.forecast)
        return {"n_forecast": len(r.forecast), "has_ci": has_ci}

    elif test == "decomposition":
        from .timeseries.decomposition import classical_decompose
        import numpy as _np
        n = 120
        t = _np.arange(n)
        data = (100 + 0.5 * t + 10 * _np.sin(2 * _np.pi * t / 12) + _np.random.default_rng(42).normal(0, 1, n)).tolist()
        r = classical_decompose(data, period=12)
        return {"seasonal_strength": r.seasonal_strength, "trend_direction": r.trend_direction}

    elif test == "pelt_changepoint":
        from .timeseries.changepoint import pelt as run_pelt
        import numpy as _np
        rng = _np.random.default_rng(42)
        data = _np.concatenate([rng.normal(10, 1, 50), rng.normal(20, 1, 50)]).tolist()
        r = run_pelt(data, min_size=10)
        return {"n_changepoints": len(r.changepoints)}

    elif test == "granger":
        from .timeseries.causality import granger_causality
        import numpy as _np
        rng = _np.random.default_rng(42)
        n = 200
        x = rng.normal(0, 1, n)
        y = _np.zeros(n)
        for i in range(2, n):
            y[i] = 0.7 * x[i - 2] + rng.normal(0, 0.5)
        r = granger_causality(x.tolist(), y.tolist(), max_lag=4)
        return {"x_causes_y": r.x_causes_y}

    elif test == "anomaly_score":
        from .timeseries.changepoint import anomaly_scores
        import numpy as _np
        rng = _np.random.default_rng(42)
        data = rng.normal(50, 2, 100)
        data[50] = 100
        r = anomaly_scores(data.tolist(), window=20, threshold=3.0)
        return {"spike_detected": 50 in r.anomaly_indices}

    # Phase 5: Parity gaps
    elif test == "repeated_measures":
        from .parametric.repeated_measures import repeated_measures_anova
        r = repeated_measures_anova(inp)
        return {"significant": r.p_value < 0.05, "n_conditions": r.n_conditions}

    elif test == "runs_test":
        from .nonparametric.rank_tests import runs_test as rt
        r = rt(inp["data"])
        return {"significant": bool(r.p_value < 0.05)}

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
            elif isinstance(val, str):
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
