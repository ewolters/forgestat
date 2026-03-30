"""Inter-rater agreement — Krippendorff's alpha, Fleiss' kappa.

Handles multiple raters, nominal/ordinal/interval scales.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class AgreementResult:
    """Inter-rater agreement result."""

    method: str  # "krippendorff" or "fleiss"
    value: float = 0.0  # agreement coefficient
    se: float | None = None
    ci_lower: float | None = None
    ci_upper: float | None = None
    n_subjects: int = 0
    n_raters: int = 0
    interpretation: str = ""  # "poor", "slight", "fair", "moderate", "substantial", "almost_perfect"


def krippendorff_alpha(
    ratings: list[list[float | str | None]],
    level: str = "nominal",
) -> AgreementResult:
    """Krippendorff's alpha for inter-rater reliability.

    Handles any number of raters, missing data, and multiple scale types.

    Args:
        ratings: n_raters × n_subjects matrix. None = missing.
        level: "nominal", "ordinal", "interval", or "ratio".

    Returns:
        AgreementResult with alpha and interpretation.
    """
    R = np.array(ratings, dtype=object)
    n_raters, n_subjects = R.shape

    # Collect all valid pairs within each subject
    observed_disagreement = 0.0
    expected_disagreement = 0.0
    total_pairs = 0
    all_values = []

    for j in range(n_subjects):
        values = [R[i, j] for i in range(n_raters) if R[i, j] is not None]
        if len(values) < 2:
            continue
        all_values.extend(values)

        n_v = len(values)
        for a in range(n_v):
            for b in range(a + 1, n_v):
                observed_disagreement += _distance(values[a], values[b], level)
                total_pairs += 1

    if total_pairs == 0:
        return AgreementResult(method="krippendorff", n_subjects=n_subjects, n_raters=n_raters)

    observed_disagreement /= total_pairs

    # Expected disagreement: over all value pairs in entire dataset
    n_all = len(all_values)
    exp_pairs = 0
    exp_sum = 0.0
    for a in range(n_all):
        for b in range(a + 1, n_all):
            exp_sum += _distance(all_values[a], all_values[b], level)
            exp_pairs += 1

    expected_disagreement = exp_sum / exp_pairs if exp_pairs > 0 else 0

    alpha = 1 - observed_disagreement / expected_disagreement if expected_disagreement > 0 else 1.0

    return AgreementResult(
        method="krippendorff",
        value=float(alpha),
        n_subjects=n_subjects,
        n_raters=n_raters,
        interpretation=_interpret_agreement(alpha),
    )


def fleiss_kappa(
    ratings_matrix: list[list[int]],
    n_raters: int,
) -> AgreementResult:
    """Fleiss' kappa for multiple raters with nominal categories.

    Args:
        ratings_matrix: n_subjects × n_categories matrix.
            Each entry = number of raters who assigned that category to that subject.
        n_raters: Total number of raters.

    Returns:
        AgreementResult with kappa and interpretation.
    """
    M = np.asarray(ratings_matrix, dtype=float)
    n_subjects, n_categories = M.shape

    # P_i: proportion of agreement for subject i
    P_i = (np.sum(M ** 2, axis=1) - n_raters) / (n_raters * (n_raters - 1))
    P_bar = float(np.mean(P_i))

    # P_j: proportion of ratings in category j
    p_j = np.sum(M, axis=0) / (n_subjects * n_raters)
    P_e = float(np.sum(p_j ** 2))

    kappa = (P_bar - P_e) / (1 - P_e) if (1 - P_e) > 0 else 1.0

    return AgreementResult(
        method="fleiss",
        value=float(kappa),
        n_subjects=n_subjects,
        n_raters=n_raters,
        interpretation=_interpret_agreement(kappa),
    )


def _distance(a, b, level: str) -> float:
    """Compute distance between two values based on measurement level."""
    if level == "nominal":
        return 0.0 if a == b else 1.0
    elif level == "ordinal":
        try:
            return abs(float(a) - float(b))
        except (ValueError, TypeError):
            return 0.0 if a == b else 1.0
    elif level in ("interval", "ratio"):
        try:
            return (float(a) - float(b)) ** 2
        except (ValueError, TypeError):
            return 0.0 if a == b else 1.0
    return 0.0 if a == b else 1.0


def _interpret_agreement(value: float) -> str:
    """Landis & Koch interpretation of agreement coefficients."""
    if value < 0:
        return "poor"
    elif value < 0.20:
        return "slight"
    elif value < 0.40:
        return "fair"
    elif value < 0.60:
        return "moderate"
    elif value < 0.80:
        return "substantial"
    else:
        return "almost_perfect"
