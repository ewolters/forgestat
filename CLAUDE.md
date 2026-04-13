# ForgeStat

General-purpose statistics engine. numpy/scipy based. No web framework, no database.

## Architecture

```
forgestat/
├── core/           # types.py (result dataclasses), assumptions.py, effect_size.py, distributions.py, sampling.py
├── parametric/     # ttest, anova, correlation, chi_square, proportion, equivalence,
│                   # repeated_measures, split_plot, variance, mixed (linear mixed-effects)
├── nonparametric/  # rank_tests (Mann-Whitney, Kruskal-Wallis, Wilcoxon, Friedman)
├── posthoc/        # comparisons (Tukey, Dunnett, Games-Howell, Dunn, Bonferroni, Scheffe)
├── regression/     # linear, logistic, nonlinear, stepwise, glm, robust, best_subsets,
│                   # regularized (Ridge, Lasso, Elastic Net)
├── bayesian/       # tests (Bayesian t, ANOVA, proportion, correlation, regression)
├── exploratory/    # univariate, multivariate, multi_vari, meta
├── power/          # sample_size (t, ANOVA, chi-square, proportion, regression, DOE)
├── quality/        # capability, acceptance, anom, variance_components, desirability
├── reliability/    # survival (Kaplan-Meier), distributions (Weibull, lognormal), cox
├── msa/            # gage_rr, agreement (Bland-Altman, linearity), kappa (ICC, Krippendorff)
├── timeseries/     # stationarity, correlation, forecasting, decomposition, changepoint, causality
├── intelligence/   # engine (auto_analyze), interpret (explain, recommend_next, validate),
│                   # corrections (BH-FDR, Holm-Bonferroni)
└── calibration.py  # Self-calibration service
```

## Running Tests

```bash
pip install -e ".[dev]"
pytest tests/ -v
```

320 tests across 14 test files.

## Key Design Decisions

- Namespace package: `__all__` lists submodule names, not individual functions
- Each submodule has its own public API; import from submodule directly
- core.types defines shared result dataclasses used across all modules
- core.assumptions provides pre-checks (normality, homogeneity) used by parametric tests
- No global imports at package level to keep startup fast

## Intelligence Engine (`intelligence/`)

The intelligence module is the "statistical reasoning" layer. It does NOT call any AI/API — it's pure Python deterministic decision logic.

### `auto_analyze(data, goal="compare", alpha=0.05)` → AnalysisResult
Give it data + intent. It profiles the data, checks assumptions, picks the right test, runs it,
computes effect sizes, runs post-hocs if significant, generates interpretation, recommends next steps.

Goals: `"compare"`, `"correlate"`, `"predict"`, `"distribute"`, `"equivalence"`

Decision path for `goal="compare"`:
- 1 group + mu → one-sample t-test (or Wilcoxon if non-normal)
- 2 groups → two-sample t-test or Mann-Whitney (based on normality/variance checks)
- 3+ groups → ANOVA or Kruskal-Wallis → auto post-hoc if significant

### `explain(result)` → Explanation
Takes any TestResult/AnovaResult/etc. and generates structured plain English.

### `recommend_next(result)` → list[Recommendation]
After any analysis: what to run next, why, and which forgestat function to call.

### `validate_conclusion(result)` → ValidationReport
Is this result trustworthy? Checks power, assumptions, sample size, borderline p-values.

### `compare_methods(data)` → ComparisonReport
Runs parametric AND non-parametric on the same data. Shows agreement/disagreement.

### `benjamini_hochberg(p_values)` → BHResult
FDR correction for multiple testing. Also: `holm_bonferroni()`.

### Integration with SVEND
In SVEND, the intelligence engine provides the deterministic test selection and routing.
Claude API provides the contextual interpretation — connecting results to the user's
FMEA, change requests, knowledge graph, and process history.

## Mixed Models (`parametric/mixed.py`)

### `mixed_model(data, response, fixed, random)` → MixedResult
Linear mixed-effects model with REML estimation. For nested manufacturing data
(measurements within parts within machines within shifts).

### `variance_decomposition(data, response, factors)` → VarianceDecomposition
Decompose total variance into between-group and within-group components with ICC.

### `nested_anova(data, response, factors)` → MixedResult
Convenience for nested ANOVA (parts within machines).

## Regularized Regression (`regression/regularized.py`)

### `ridge(X, y, alpha)` → RegularizedResult
L2 penalty. GCV for auto-alpha selection.

### `lasso(X, y, alpha)` → RegularizedResult
L1 penalty. Coordinate descent. Produces sparse solutions.

### `elastic_net(X, y, alpha, l1_ratio)` → RegularizedResult
Combined L1+L2. `l1_ratio=0` is Ridge, `l1_ratio=1` is Lasso.

### `regularization_path(X, y, method, n_alphas)` → PathResult
Coefficient paths across alpha range for visualization.

## Dependencies

- numpy, scipy (required)
- statsmodels (optional — regression, GLM)
