# ForgeStat

General-purpose statistics engine. Pure Python computation with numpy/scipy.

## Install

```bash
pip install forgestat
```

## Quick Start

```python
from forgestat.parametric.ttest import one_sample_t, two_sample_t
from forgestat.nonparametric.rank_tests import mann_whitney_u
from forgestat.regression.linear import ols_regression
from forgestat.posthoc.comparisons import tukey_hsd

result = two_sample_t([1, 2, 3, 4], [2, 3, 4, 5])
mw = mann_whitney_u([1, 2, 3], [4, 5, 6])
```

## Modules

| Module | Contents |
|---|---|
| `core` | Result types, assumption checks, effect sizes, distribution fitting |
| `parametric` | t-tests, ANOVA, correlation, chi-square, proportions, equivalence, repeated measures, split-plot, variance tests |
| `nonparametric` | Mann-Whitney, Kruskal-Wallis, Wilcoxon, Friedman, runs test |
| `posthoc` | Tukey HSD, Dunnett, Games-Howell, Dunn, Bonferroni, Scheffe |
| `regression` | OLS, logistic, Poisson, nonlinear, stepwise, GLM, robust, best subsets, ordinal, orthogonal |
| `bayesian` | Bayesian t-test, ANOVA, proportion, correlation, regression |
| `exploratory` | Descriptive stats, PCA, MANOVA, meta-analysis, multi-vari |
| `power` | Sample size for t, ANOVA, chi-square, proportion, regression, variance, DOE, tolerance |
| `quality` | Process capability, acceptance sampling, ANOM, variance components |
| `reliability` | Kaplan-Meier, Weibull, Cox PH, exponential, lognormal |
| `msa` | Gage R&R, ICC, Bland-Altman, linearity/bias, Krippendorff alpha |
| `timeseries` | ADF, KPSS, ACF/PACF, ARIMA, SARIMA, decomposition, changepoint, Granger causality |

## Dependencies

- numpy, scipy (required)
- statsmodels (optional, for select advanced methods)

## License

MIT
