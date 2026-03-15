"""Gap analysis and actionable recommendation engine.

Given an EvaluationResult, identifies factors where the applicant scores
below the gap threshold and maps each to a concrete, actionable
recommendation with a priority level.
"""

from __future__ import annotations

from dataclasses import dataclass
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
    # ── Math dimension ───────────────────────────────────────────────
    "calculus_series": (
        "Complete Calculus I-III including multivariable calculus"
        " (required by all programs)"
    ),
    "linear_algebra": (
        "Take a proof-based Linear Algebra course"
        " (upper-division; eigenvalues, vector spaces, linear transformations)"
    ),
    "probability": (
        "Take calculus-based Probability Theory"
        " (distributions, CLT, conditional expectation)"
    ),
    "ode_pde": (
        "Take ODE (required); PDE is a differentiator for top-5"
        " (boundary value problems, Fourier methods)"
    ),
    "real_analysis": (
        "Take Real Analysis / Real Variables"
        " (required by Princeton, NYU Courant; strongly recommended by CMU, Baruch)"
    ),
    "numerical_analysis": (
        "Take Numerical Analysis / Numerical Methods"
        " (required by Berkeley; interpolation, quadrature, solving DEs)"
    ),
    "stochastic_processes": (
        "Take Stochastic Processes"
        " (Markov chains, Poisson processes, random walks)"
    ),
    "stochastic_calculus": (
        "Take Stochastic Calculus / Itô Calculus"
        " (Brownian motion, SDEs — top differentiator for competitive applicants)"
    ),
    "optimization": (
        "Take Optimization (linear/convex/nonlinear programming)"
        " (valuable for Cornell ORIE, GaTech, portfolio optimization)"
    ),
    # ── Statistics dimension ─────────────────────────────────────────
    "math_stats": (
        "Take Mathematical Statistics at the 400+ level"
        " (MLE, hypothesis testing, sufficient statistics)"
    ),
    "regression": (
        "Take Regression Analysis / Applied Statistics"
        " (linear models, ANOVA, diagnostics)"
    ),
    "time_series": (
        "Take Time Series Analysis"
        " (ARIMA, GARCH — essential for financial data modeling)"
    ),
    "econometrics": "Take Econometrics (IV, panel data, causal inference)",
    "stat_learning_ml": (
        "Take Statistical Learning or Machine Learning"
        " (regularization, cross-validation, tree methods)"
    ),
    "stat_computing": (
        "Take Statistical Computing in R or Python"
        " (simulation, bootstrapping, computational methods)"
    ),
    "courses_400_level_count": (
        "Enroll in more upper-division (400+) statistics courses"
        " to demonstrate depth on your transcript"
    ),
    # ── CS dimension ─────────────────────────────────────────────────
    "cpp_proficiency": (
        "Take a C++ programming course"
        " (essential for Baruch, CMU, Berkeley; tested in interviews)"
    ),
    "python_proficiency": (
        "Take Python programming or build substantial Python projects"
        " (NumPy, pandas, data analysis pipelines)"
    ),
    "data_structures_algo": (
        "Take Data Structures and Algorithms"
        " (complexity analysis, trees, graphs, dynamic programming)"
    ),
    "ml_course": (
        "Take a Machine Learning course"
        " (neural networks, ensemble methods — increasingly valued)"
    ),
    "numerical_computing": "Take Numerical or Scientific Computing",
    "software_engineering": (
        "Take Software Engineering or OOP"
        " (design patterns, version control, testing — valued by industry)"
    ),
    "is_cs_major": "Consider a CS minor or additional CS electives",
    # ── Finance/Econ dimension ───────────────────────────────────────
    "micro_macro": (
        "Take Intermediate Micro and Macro Economics"
        " (utility theory, equilibrium, monetary policy)"
    ),
    "investments_finance": (
        "Take Investments or Corporate Finance"
        " (CAPM, efficient markets, valuation)"
    ),
    "derivatives": (
        "Take a Derivatives / Options Pricing course"
        " (Black-Scholes, binomial models, Greeks)"
    ),
    "portfolio_theory": (
        "Take Portfolio Theory / Asset Pricing"
        " (mean-variance optimization, factor models)"
    ),
    "risk_management": (
        "Take Risk Management (VaR, stress testing, credit risk)"
    ),
    "financial_econometrics": "Take Financial Econometrics",
    "game_theory": (
        "Take Game Theory (lower priority for most MFE programs)"
    ),
    # ── GPA dimension ────────────────────────────────────────────────
    "cumulative_gpa": (
        "Focus on raising GPA by excelling in remaining courses"
    ),
    "quant_gpa": (
        "Prioritize high grades in quant courses"
        " (math, stats, CS) to boost quant GPA"
    ),
    "trend": (
        "Take challenging 400/500-level courses"
        " and earn strong grades to show upward trend"
    ),
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

        fallback = (
            f"Improve your {factor.replace('_', ' ')} background"
            " with additional coursework or projects"
        )
        action = _FACTOR_ACTIONS.get(factor, fallback)
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
