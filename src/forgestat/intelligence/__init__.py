"""Statistical Intelligence Engine — automated analysis, interpretation, and recommendations.

Deterministic decision logic. No AI/API calls.
Takes data + intent, selects and runs the right analysis, explains the result.
"""

from .engine import auto_analyze, profile_data
from .interpret import compare_methods, explain, recommend_next, validate_conclusion
from .corrections import benjamini_hochberg, fdr_summary, holm_bonferroni

__all__ = [
    "auto_analyze",
    "compare_methods",
    "explain",
    "profile_data",
    "recommend_next",
    "validate_conclusion",
    "benjamini_hochberg",
    "holm_bonferroni",
    "fdr_summary",
]
