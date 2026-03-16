"""Course optimization engine.

Recommends which courses to take next to maximize profile improvement
across the five evaluation dimensions (math, statistics, CS,
finance/economics, GPA).

The optimizer works by:
    1. Examining every course category in the taxonomy.
    2. Computing each category's potential impact based on the dimension
       weight, factor weight, and the gap between the current score and
       the 9.0 ceiling.
    3. Adding a bonus for categories that are required prerequisites at
       the user's target programmes but not yet satisfied.
    4. Returning the top-N highest-impact recommendations.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .models import CourseCategory, ProgramData, UserProfile
from .profile_evaluator import (
    DIMENSION_WEIGHTS,
    _best_score_for_categories,
    _best_score_for_category,
)

# ===================================================================
# Result dataclass
# ===================================================================


@dataclass
class CourseRecommendation:
    """A single course recommendation with its computed impact."""

    category: str  # e.g. "stochastic_calculus"
    dimension: str  # e.g. "math"
    impact_score: float  # higher = more impactful
    reason: str  # human-readable explanation
    prereq_coverage: int  # how many target programs require this as a prereq


# ===================================================================
# Factor-to-category mapping
# ===================================================================

# Each entry maps (dimension, factor_name, factor_weight) -> set of
# course categories that contribute to that factor's score.
#
# Extracted from the scorer functions in profile_evaluator.py.

def _f(dim: str, factor: str, weight: float, cats: set[str]) -> dict[str, Any]:
    return {"dimension": dim, "factor": factor, "weight": weight, "categories": cats}


_FACTOR_CATEGORY_MAP: list[dict[str, Any]] = [
    # ── Math dimension (weight 0.30) ──
    _f("math", "calculus_series", 0.15, {"calculus"}),
    _f("math", "linear_algebra", 0.15, {"linear_algebra"}),
    _f("math", "probability", 0.15, {"probability"}),
    _f("math", "ode_pde", 0.10, {"ode", "pde"}),
    _f("math", "real_analysis", 0.12, {"real_analysis"}),
    _f("math", "numerical_analysis", 0.08, {"numerical_analysis"}),
    _f("math", "stochastic_processes", 0.08, {"stochastic_processes"}),
    _f("math", "stochastic_calculus", 0.10, {"stochastic_calculus"}),
    _f("math", "optimization", 0.07, {"optimization"}),
    # ── Statistics dimension (weight 0.20) ──
    _f("statistics", "math_stats", 0.22, {"statistics"}),
    _f("statistics", "regression", 0.13, {"regression"}),
    _f("statistics", "time_series", 0.18, {"time_series"}),
    _f("statistics", "econometrics", 0.12, {"econometrics"}),
    _f("statistics", "stat_learning_ml", 0.13, {"stat_learning", "machine_learning"}),
    _f("statistics", "stat_computing", 0.10, {"stat_computing"}),
    # ── CS dimension (weight 0.20) ──
    _f("cs", "cpp_proficiency", 0.25, {"programming_cpp"}),
    _f("cs", "python_proficiency", 0.20, {"programming_python"}),
    _f("cs", "data_structures_algo", 0.18, {"data_structures", "algorithms"}),
    _f("cs", "ml_course", 0.10, {"machine_learning"}),
    _f("cs", "numerical_computing", 0.07, {"numerical_analysis"}),
    _f("cs", "software_engineering", 0.10, {"software_engineering"}),
    # ── Finance / Econ dimension (weight 0.15) ──
    _f("finance_econ", "micro_macro", 0.18, {"microeconomics", "macroeconomics"}),
    _f("finance_econ", "investments_finance", 0.20, {"finance"}),
    _f("finance_econ", "derivatives", 0.18, {"derivatives"}),
    _f("finance_econ", "portfolio_theory", 0.10, {"portfolio_theory"}),
    _f("finance_econ", "risk_management", 0.12, {"risk_management"}),
    _f("finance_econ", "financial_econometrics", 0.12, {"financial_econometrics"}),
    _f("finance_econ", "game_theory", 0.10, {"game_theory"}),
]

# Prereq category expansion (mirrors prerequisite_matcher._CATEGORY_MAP).
# Used to check whether a user's courses satisfy a programme prerequisite.
_PREREQ_CATEGORY_MAP: dict[str, set[str]] = {
    "calculus": {"calculus"},
    "linear_algebra": {"linear_algebra"},
    "probability": {"probability"},
    "statistics": {"statistics"},
    "differential_equations": {"ode", "pde"},
    "ode": {"ode"},
    "pde": {"pde"},
    "real_analysis": {"real_analysis"},
    "numerical_methods": {"numerical_analysis"},
    "numerical_analysis": {"numerical_analysis"},
    "stochastic_calculus": {"stochastic_calculus", "stochastic_processes"},
    "stochastic_processes": {"stochastic_processes"},
    "econometrics": {"econometrics", "financial_econometrics"},
    "time_series": {"time_series"},
    "stat_computing": {"stat_computing"},
    "stat_learning": {"stat_learning", "machine_learning"},
    "programming": {"programming_cpp", "programming_python"},
    "programming_cpp": {"programming_cpp"},
    "programming_python": {"programming_python"},
    "data_structures": {"data_structures"},
    "algorithms": {"algorithms"},
    "machine_learning": {"machine_learning", "stat_learning"},
    "finance": {"finance"},
    "microeconomics": {"microeconomics"},
    "macroeconomics": {"macroeconomics"},
    "risk_management": {"risk_management"},
    "financial_econometrics": {"financial_econometrics"},
    "game_theory": {"game_theory"},
    "partial_differential_equations": {"pde"},
    "derivatives": {"derivatives"},
    "portfolio_theory": {"portfolio_theory"},
    "optimization": {"optimization"},
    "software_engineering": {"software_engineering"},
    "regression": {"regression"},
}

# Human-readable labels for course categories.
_CATEGORY_LABELS: dict[str, str] = {
    "calculus": "Calculus (Calc I-III)",
    "linear_algebra": "Linear Algebra",
    "probability": "Probability Theory",
    "ode": "Ordinary Differential Equations",
    "pde": "Partial Differential Equations",
    "real_analysis": "Real Analysis",
    "numerical_analysis": "Numerical Analysis / Methods",
    "stochastic_processes": "Stochastic Processes",
    "stochastic_calculus": "Stochastic Calculus / Ito Calculus",
    "optimization": "Optimization",
    "statistics": "Mathematical Statistics",
    "regression": "Regression / Applied Statistics",
    "time_series": "Time Series Analysis",
    "econometrics": "Econometrics",
    "stat_learning": "Statistical Learning",
    "machine_learning": "Machine Learning",
    "stat_computing": "Statistical Computing",
    "programming_cpp": "C++ Programming",
    "programming_python": "Python Programming",
    "data_structures": "Data Structures",
    "algorithms": "Algorithms",
    "software_engineering": "Software Engineering / OOP",
    "finance": "Investments / Corporate Finance",
    "derivatives": "Derivatives / Options Pricing",
    "portfolio_theory": "Portfolio Theory / Asset Pricing",
    "microeconomics": "Intermediate Microeconomics",
    "macroeconomics": "Intermediate Macroeconomics",
    "risk_management": "Risk Management",
    "financial_econometrics": "Financial Econometrics",
    "game_theory": "Game Theory",
}

# Score ceiling: categories at or above this level are not recommended.
_SCORE_CEILING = 9.0

# Prereq bonus per programme that requires this category.
_PREREQ_BONUS = 0.5


# ===================================================================
# Internal helpers
# ===================================================================


def _current_score_for_categories(
    profile: UserProfile,
    categories: set[str],
) -> float:
    """Return the user's best score across courses in *categories*."""
    return _best_score_for_categories(profile.coursework, categories)


def _user_has_category(profile: UserProfile, category: str) -> bool:
    """Return True if the user has at least one course in *category*."""
    return any(c.category == category for c in profile.coursework)


def _user_satisfies_prereq_category(
    profile: UserProfile,
    prereq_category: str,
) -> bool:
    """Return True if the user has coursework that satisfies a
    programme prerequisite *prereq_category*.
    """
    acceptable = _PREREQ_CATEGORY_MAP.get(prereq_category, {prereq_category})
    return any(c.category in acceptable for c in profile.coursework)


def _count_prereq_coverage(
    category: str,
    programs: list[ProgramData],
    profile: UserProfile,
) -> int:
    """Count how many target programmes require *category* as a
    prerequisite that the user does NOT yet satisfy.

    A programme counts if it has *category* (or a broader alias that
    maps to *category*) in its required prerequisites and the user
    is missing that prerequisite.
    """
    count = 0
    for program in programs:
        for prereq in program.prerequisites_required:
            # Expand the prereq category to the set of course categories
            # that would satisfy it.
            acceptable = _PREREQ_CATEGORY_MAP.get(
                prereq.category, {prereq.category}
            )
            if category in acceptable:
                # Does the user already satisfy this prereq?
                if not _user_satisfies_prereq_category(profile, prereq.category):
                    count += 1
                    break  # count each program at most once per category
    return count


def _build_reason(
    category: str,
    dimension: str,
    current_score: float,
    prereq_coverage: int,
) -> str:
    """Build a human-readable reason string for a recommendation."""
    label = _CATEGORY_LABELS.get(category, category.replace("_", " ").title())

    parts: list[str] = []
    if current_score == 0:
        parts.append(f"{label} is missing from your coursework")
    else:
        parts.append(
            f"{label} score is {current_score:.1f}/10 in the {dimension} dimension"
        )

    if prereq_coverage > 0:
        prog_word = "program" if prereq_coverage == 1 else "programs"
        parts.append(
            f"required by {prereq_coverage} target {prog_word} you haven't covered"
        )

    return "; ".join(parts)


# ===================================================================
# Public API
# ===================================================================


def optimize_courses(
    profile: UserProfile,
    programs: list[ProgramData],
    max_courses: int = 3,
) -> list[CourseRecommendation]:
    """Recommend courses that would most improve the user's profile.

    For each course category in the taxonomy (excluding ``other``):

    1. Skip if the user already scores >= 9.0 in that category.
    2. Compute **impact** = ``dim_weight * factor_weight * (9.0 - current_score)``
       for every factor that maps to this category, summed across all
       such factors.
    3. Compute **prereq_coverage** = count of target programmes that
       require this category as a prerequisite and the user hasn't
       satisfied it yet.
    4. Add a prereq bonus: ``prereq_coverage * 0.5`` to the impact.
    5. Generate a reason string explaining why this course helps.

    Parameters
    ----------
    profile:
        The applicant's current profile.
    programs:
        List of target programmes (used for prereq coverage).
    max_courses:
        Maximum number of recommendations to return.

    Returns
    -------
    list[CourseRecommendation]
        Sorted by ``impact_score`` descending, capped at *max_courses*.
    """
    # Gather all course categories except OTHER.
    all_categories = {cat.value for cat in CourseCategory if cat != CourseCategory.OTHER}

    # Build per-category impact scores.
    recommendations: list[CourseRecommendation] = []

    for category in sorted(all_categories):
        # Current best score for this category.
        current_score = _best_score_for_category(profile.coursework, category)

        # Skip categories where the user is already strong.
        if current_score >= _SCORE_CEILING:
            continue

        # Accumulate impact from every factor that includes this category.
        impact = 0.0
        dimension = ""
        for fmap in _FACTOR_CATEGORY_MAP:
            if category in fmap["categories"]:
                dim_weight = DIMENSION_WEIGHTS[fmap["dimension"]]
                factor_weight = fmap["weight"]
                gap = _SCORE_CEILING - current_score
                impact += dim_weight * factor_weight * gap
                # Use the dimension of the highest-weight contributing factor.
                if not dimension:
                    dimension = fmap["dimension"]

        # Skip categories that don't map to any factor (e.g. bayesian,
        # fixed_income, accounting, programming_r, database).
        if impact == 0.0:
            continue

        # Prereq coverage bonus.
        prereq_coverage = _count_prereq_coverage(category, programs, profile)
        impact += prereq_coverage * _PREREQ_BONUS

        reason = _build_reason(category, dimension, current_score, prereq_coverage)

        recommendations.append(
            CourseRecommendation(
                category=category,
                dimension=dimension,
                impact_score=round(impact, 4),
                reason=reason,
                prereq_coverage=prereq_coverage,
            )
        )

    # Sort by impact descending, then alphabetically for ties.
    recommendations.sort(key=lambda r: (-r.impact_score, r.category))

    return recommendations[:max_courses]
