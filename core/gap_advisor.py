"""Gap analysis and actionable recommendation engine.

Given an EvaluationResult, identifies factors where the applicant scores
below the gap threshold and maps each to a concrete, actionable
recommendation with a priority level.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


# ===================================================================
# Recommendation data
# ===================================================================

@dataclass
class GapRecommendation:
    """A single gap with its recommended action."""

    factor: str
    dimension: str
    score: float
    priority: str  # "High", "Medium", "Low"
    action: str


# ===================================================================
# Factor-to-action mapping
# ===================================================================

# Maps factor names (as produced by profile_evaluator) to actionable
# course/activity suggestions.  When a factor is missing from this dict
# the advisor falls back to a generic message.

_FACTOR_ACTIONS: dict[str, str] = {
    # Math dimension
    "calculus_series":      "Take Calculus III or equivalent (multivariable calculus is required by nearly all programs)",
    "linear_algebra":       "Take a Linear Algebra course with proof-based content (MATH 415 level or above)",
    "probability":          "Take a calculus-based Probability course (e.g., STAT 400 or equivalent)",
    "ode_pde":              "Take Differential Equations (ODE); PDE is a strong plus for top-5 programs",
    "real_analysis":        "Take Real Analysis / Real Variables (strongly recommended for CMU, Baruch, Princeton)",
    "numerical_analysis":   "Take Numerical Analysis or Numerical Methods (CS 450 / MATH 450 level)",
    "stochastic_processes": "Take Stochastic Calculus or Stochastic Processes (recommended for top-5 programs)",

    # Statistics dimension
    "math_stats":             "Take Mathematical Statistics at the 400 level (e.g., STAT 410)",
    "time_series":            "Take Time Series Analysis (STAT 429 or equivalent)",
    "econometrics":           "Take an Econometrics course (ECON 480 or equivalent)",
    "stat_learning_ml":       "Take Statistical Learning or Machine Learning (STAT 432 / CS 446)",
    "stat_computing":         "Take Statistical Computing (STAT 428 / R or Python-based)",
    "courses_400_level_count": "Enroll in more 400-level statistics courses to strengthen your transcript",

    # CS dimension
    "cpp_proficiency":      "Take a C++ programming course (essential for Baruch, CMU, and most top programs)",
    "python_proficiency":   "Take a Python programming course or complete substantial Python projects",
    "data_structures_algo": "Take Data Structures and Algorithms (CS 225 + CS 374 or equivalent)",
    "ml_course":            "Take a Machine Learning course (CS 446 / STAT 432 level)",
    "numerical_computing":  "Take Numerical Computing or Scientific Computing (CS 450 level)",
    "is_cs_major":          "Consider adding a CS minor or taking additional CS electives to strengthen your profile",

    # Finance/Econ dimension
    "micro_macro":            "Take Intermediate Micro and Macro Economics courses",
    "investments_finance":    "Take an Investments or Corporate Finance course",
    "derivatives":            "Take a Derivatives/Options Pricing course or Financial Risk Management",
    "risk_management":        "Take a Risk Management or Financial Risk course",
    "financial_econometrics": "Take Financial Econometrics (ECON 472 or equivalent)",
    "game_theory":            "Take Game Theory (useful but lower priority for most MFE programs)",

    # GPA dimension
    "cumulative_gpa": "Focus on raising cumulative GPA by excelling in remaining coursework",
    "quant_gpa":      "Prioritize high grades in quantitative courses (math, stats, CS) to boost quant GPA",
    "trend":          "Take challenging 400/500-level courses and earn strong grades to show upward trend",
}


def _priority_for_score(score: float) -> str:
    """Assign a priority level based on the gap score.

    - High:   score == 0 (factor is completely missing)
    - Medium: score < 4  (significant weakness)
    - Low:    score < 6  (below the gap threshold but present)
    """
    if score == 0:
        return "High"
    if score < 4:
        return "Medium"
    return "Low"


# ===================================================================
# Public API
# ===================================================================

def analyze_gaps(
    gaps: list[dict[str, Any]],
) -> list[GapRecommendation]:
    """Produce actionable recommendations for every gap.

    Parameters
    ----------
    gaps:
        The ``gaps`` list from an :class:`EvaluationResult`.  Each entry
        is a dict with keys ``dimension``, ``factor``, ``score``.

    Returns
    -------
    list[GapRecommendation]
        Sorted by priority (High first) then by score ascending.
    """
    recommendations: list[GapRecommendation] = []

    for gap in gaps:
        factor = gap.get("factor", "")
        dimension = gap.get("dimension", "")
        score = gap.get("score", 0.0)

        action = _FACTOR_ACTIONS.get(
            factor,
            f"Improve your {factor.replace('_', ' ')} background with additional coursework or projects",
        )
        priority = _priority_for_score(score)

        recommendations.append(
            GapRecommendation(
                factor=factor,
                dimension=dimension,
                score=score,
                priority=priority,
                action=action,
            )
        )

    # Sort: High > Medium > Low, then by score ascending within each tier.
    priority_order = {"High": 0, "Medium": 1, "Low": 2}
    recommendations.sort(key=lambda r: (priority_order[r.priority], r.score))

    return recommendations
