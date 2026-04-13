"""Multiple testing corrections — Benjamini-Hochberg FDR, Holm-Bonferroni.

Pure computation. No external dependencies beyond numpy.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class BHResult:
    """Benjamini-Hochberg FDR correction result."""

    original_p: list[float] = field(default_factory=list)
    adjusted_p: list[float] = field(default_factory=list)
    significant: list[bool] = field(default_factory=list)
    n_discoveries: int = 0
    fdr_level: float = 0.05


@dataclass
class CorrectionResult:
    """Generic multiple testing correction result."""

    original_p: list[float] = field(default_factory=list)
    adjusted_p: list[float] = field(default_factory=list)
    significant: list[bool] = field(default_factory=list)
    n_discoveries: int = 0
    method: str = ""
    alpha: float = 0.05


def benjamini_hochberg(p_values: list[float], alpha: float = 0.05) -> BHResult:
    """Benjamini-Hochberg procedure for controlling False Discovery Rate.

    Step-up procedure: sort p-values, compare each p(i) to i*alpha/m.

    Args:
        p_values: List of unadjusted p-values.
        alpha: Desired FDR level.

    Returns:
        BHResult with adjusted p-values and significance flags.
    """
    m = len(p_values)
    if m == 0:
        return BHResult(fdr_level=alpha)

    p_arr = np.asarray(p_values, dtype=float)

    # Sort indices by p-value
    order = np.argsort(p_arr)
    ranks = np.empty(m, dtype=int)
    ranks[order] = np.arange(1, m + 1)

    # Adjusted p-values: p_adj(i) = p(i) * m / rank(i), then enforce monotonicity
    adjusted = np.minimum(p_arr * m / ranks, 1.0)

    # Enforce monotonicity from the bottom up (in sorted order)
    # Walk from largest rank down, ensuring adjusted[i] >= adjusted[i+1]
    sorted_adj = adjusted[order].copy()
    for i in range(m - 2, -1, -1):
        sorted_adj[i] = min(sorted_adj[i], sorted_adj[i + 1])

    # Map back to original order
    final_adj = np.empty(m, dtype=float)
    final_adj[order] = sorted_adj

    significant = [bool(p < alpha) for p in final_adj]
    n_discoveries = sum(significant)

    return BHResult(
        original_p=p_values,
        adjusted_p=final_adj.tolist(),
        significant=significant,
        n_discoveries=n_discoveries,
        fdr_level=alpha,
    )


def holm_bonferroni(p_values: list[float], alpha: float = 0.05) -> CorrectionResult:
    """Holm-Bonferroni step-down procedure.

    Less conservative than Bonferroni, controls family-wise error rate.
    Sort p-values ascending. For each p(i), compare to alpha / (m - i + 1).
    Reject until the first non-rejected hypothesis; stop there.

    Args:
        p_values: List of unadjusted p-values.
        alpha: Family-wise error rate.

    Returns:
        CorrectionResult with adjusted p-values and significance flags.
    """
    m = len(p_values)
    if m == 0:
        return CorrectionResult(method="holm_bonferroni", alpha=alpha)

    p_arr = np.asarray(p_values, dtype=float)

    # Sort by p-value
    order = np.argsort(p_arr)
    sorted_p = p_arr[order]

    # Adjusted p-values: p_adj(i) = max(p(j) * (m - j)) for j <= i
    adjusted_sorted = np.empty(m, dtype=float)
    for i in range(m):
        adjusted_sorted[i] = sorted_p[i] * (m - i)

    # Enforce monotonicity (step-down: adjusted can only increase)
    for i in range(1, m):
        adjusted_sorted[i] = max(adjusted_sorted[i], adjusted_sorted[i - 1])

    # Cap at 1.0
    adjusted_sorted = np.minimum(adjusted_sorted, 1.0)

    # Map back to original order
    final_adj = np.empty(m, dtype=float)
    final_adj[order] = adjusted_sorted

    significant = [bool(p < alpha) for p in final_adj]
    n_discoveries = sum(significant)

    return CorrectionResult(
        original_p=p_values,
        adjusted_p=final_adj.tolist(),
        significant=significant,
        n_discoveries=n_discoveries,
        method="holm_bonferroni",
        alpha=alpha,
    )


def fdr_summary(tests: list[dict], alpha: float = 0.05) -> str:
    """Format a summary showing which tests survive FDR correction.

    Args:
        tests: List of {name: str, p_value: float} dicts.
        alpha: FDR level.

    Returns:
        Formatted string summary.
    """
    if not tests:
        return "No tests to summarize."

    names = [t["name"] for t in tests]
    p_values = [t["p_value"] for t in tests]

    bh = benjamini_hochberg(p_values, alpha=alpha)

    lines = [f"FDR Correction Summary (Benjamini-Hochberg, alpha={alpha})"]
    lines.append("-" * 60)
    lines.append(f"{'Test':<30} {'p-value':>10} {'adj. p':>10} {'Sig?':>6}")
    lines.append("-" * 60)

    for name, orig, adj, sig in zip(names, bh.original_p, bh.adjusted_p, bh.significant):
        marker = "  *" if sig else ""
        lines.append(f"{name:<30} {orig:>10.4f} {adj:>10.4f} {marker:>6}")

    lines.append("-" * 60)
    lines.append(f"Discoveries: {bh.n_discoveries}/{len(tests)}")

    return "\n".join(lines)
