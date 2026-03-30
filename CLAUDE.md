# ForgeStat

General-purpose statistics engine. numpy/scipy based. No web framework, no database.

## Architecture

```
forgestat/
├── core/           # types.py (result dataclasses), assumptions.py, effect_size.py, distributions.py
├── parametric/     # ttest, anova, correlation, chi_square, proportion, equivalence,
│                   # repeated_measures, split_plot, variance
├── nonparametric/  # rank_tests (Mann-Whitney, Kruskal-Wallis, Wilcoxon, Friedman)
├── posthoc/        # comparisons (Tukey, Dunnett, Games-Howell, Dunn, Bonferroni, Scheffe)
├── regression/     # linear, logistic, nonlinear, stepwise, glm, robust, best_subsets
├── bayesian/       # tests (Bayesian t, ANOVA, proportion, correlation, regression)
├── exploratory/    # univariate, multivariate, multi_vari, meta
├── power/          # sample_size (t, ANOVA, chi-square, proportion, regression, DOE)
├── quality/        # capability, acceptance, anom, variance_components
├── reliability/    # survival (Kaplan-Meier), distributions (Weibull, lognormal), cox
├── msa/            # gage_rr, agreement (Bland-Altman, linearity), kappa (ICC, Krippendorff)
├── timeseries/     # stationarity, correlation, forecasting, decomposition, changepoint, causality
└── calibration.py  # Self-calibration service
```

## Running Tests

```bash
pip install -e ".[dev]"
pytest tests/ -v
```

215 tests across 11 test files.

## Key Design Decisions

- Namespace package: `__all__` lists submodule names, not individual functions
- Each submodule has its own public API; import from submodule directly
- core.types defines shared result dataclasses used across all modules
- core.assumptions provides pre-checks (normality, homogeneity) used by parametric tests
- No global imports at package level to keep startup fast

## Dependencies

- numpy, scipy (required)
- statsmodels (optional)
