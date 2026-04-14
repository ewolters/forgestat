"""ForgeStat — General-purpose statistics engine.

Pure Python computation. No web framework, no database.

Modules:
    core        — result types, assumption checks, effect sizes, distribution fitting
    parametric  — t-tests, ANOVA, correlation, chi-square, equivalence
    nonparametric — Mann-Whitney, Kruskal-Wallis, Wilcoxon, Friedman, Mood's median
    posthoc     — Tukey HSD, Dunnett, Games-Howell, Dunn, Scheffé, Bonferroni
    conformal   — split conformal prediction (regression + classification)
    sequential  — anytime-valid inference (e-processes, confidence sequences)
    monitoring  — concept drift detection (ADWIN, Page-Hinkley, PSI)
"""

__version__ = "0.1.0"

__all__ = [
    "core",
    "parametric",
    "nonparametric",
    "posthoc",
    "regression",
    "bayesian",
    "exploratory",
    "power",
    "quality",
    "reliability",
    "msa",
    "timeseries",
    "intelligence",
    "conformal",
    "sequential",
    "monitoring",
]
