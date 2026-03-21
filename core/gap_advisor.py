"""Gap analysis and actionable recommendation engine.

Given an EvaluationResult, identifies factors where the applicant scores
below the gap threshold and maps each to a concrete, actionable
recommendation with a priority level.

Also provides program_gaps() for per-program specific gap analysis showing
prerequisite mismatches, GPA gap, and P(admit) impact estimates.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .models import EvaluationResult, ProgramData, UserProfile

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


# ===================================================================
# Per-program gap analysis
# ===================================================================


@dataclass
class ProgramGapItem:
    """A single gap item specific to a target program."""

    kind: str       # "missing_prereq" | "grade_warning" | "gpa_gap" | "profile"
    label: str      # human-readable label
    detail: str     # specific action or explanation
    severity: str   # "Critical" | "High" | "Medium" | "Low"


@dataclass
class ProgramGapReport:
    """Full per-program gap report comparing user profile vs one program."""

    program_id: str
    program_name: str
    university: str
    prereq_match_score: float
    gpa_gap: float            # user_gpa - program_avg_gpa (negative = below avg)
    admission_prob: float | None
    prob_low: float | None
    prob_high: float | None
    items: list[ProgramGapItem] = field(default_factory=list)

    @property
    def n_critical(self) -> int:
        return sum(1 for i in self.items if i.severity == "Critical")

    @property
    def n_high(self) -> int:
        return sum(1 for i in self.items if i.severity == "High")


def program_gaps(
    profile: "UserProfile",
    program: "ProgramData",
    eval_result: "EvaluationResult | None" = None,
) -> ProgramGapReport:
    """Generate a program-specific gap report.

    Combines prerequisite matching, GPA comparison, and LR admission
    probability to give targeted advice for a single program.

    Parameters
    ----------
    profile:
        The applicant's profile.
    program:
        The target program data.
    eval_result:
        Optional pre-computed evaluation (avoids recomputation).

    Returns
    -------
    ProgramGapReport
        Detailed gap items ordered by severity.
    """
    from .lr_predictor import predict_prob_full
    from .prerequisite_matcher import match_prerequisites

    pmatch = match_prerequisites(profile, program)
    items: list[ProgramGapItem] = []

    # --- Prerequisite gaps -------------------------------------------
    for missing_cat in pmatch.missing:
        cat_label = missing_cat.replace("_", " ").title()
        action = _FACTOR_ACTIONS.get(missing_cat)
        if not action:
            # Try to find by partial key match
            for key, val in _FACTOR_ACTIONS.items():
                if missing_cat in key or key in missing_cat:
                    action = val
                    break
        detail = action or f"Take a {cat_label} course (required by {program.name})"
        severity = "Critical" if pmatch.match_score < 0.5 else "High"
        items.append(
            ProgramGapItem(
                kind="missing_prereq",
                label=f"Missing: {cat_label}",
                detail=detail,
                severity=severity,
            )
        )

    # --- Grade warnings ----------------------------------------------
    for warning in pmatch.warnings:
        items.append(
            ProgramGapItem(
                kind="grade_warning",
                label="Grade below threshold",
                detail=warning,
                severity="Medium",
            )
        )

    # --- GPA gap -----------------------------------------------------
    gpa_gap = profile.gpa - (program.avg_gpa or 3.80)
    if gpa_gap < -0.20:
        items.append(
            ProgramGapItem(
                kind="gpa_gap",
                label=f"GPA gap: {profile.gpa:.2f} vs avg {program.avg_gpa:.2f}",
                detail=(
                    f"Your GPA is {abs(gpa_gap):.2f} below the program average. "
                    "Maximize grades in remaining courses and emphasize quant GPA."
                ),
                severity="High",
            )
        )
    elif gpa_gap < -0.05:
        items.append(
            ProgramGapItem(
                kind="gpa_gap",
                label=f"GPA slightly below avg: {profile.gpa:.2f} vs {program.avg_gpa:.2f}",
                detail="Borderline GPA — strengthen other application components.",
                severity="Medium",
            )
        )

    # --- Profile-level weakness (from eval_result) -------------------
    if eval_result is not None:
        for gap in eval_result.gaps:
            factor = gap.get("factor", "")
            score = gap.get("score", 0.0)
            # Only flag Critical profile gaps (completely missing)
            if score == 0 and factor in {
                "stochastic_calculus", "real_analysis", "cpp_proficiency"
            }:
                label = factor.replace("_", " ").title()
                action = _FACTOR_ACTIONS.get(factor, f"Take {label}")
                items.append(
                    ProgramGapItem(
                        kind="profile",
                        label=f"Profile gap: {label} missing",
                        detail=action,
                        severity="High",
                    )
                )

    # --- Admission probability ---------------------------------------
    lr_pred = predict_prob_full(
        program.id, profile.gpa, profile.test_scores.gre_quant, profile
    )
    admission_prob = lr_pred.prob if lr_pred is not None else None
    prob_low = lr_pred.prob_low if lr_pred is not None else None
    prob_high = lr_pred.prob_high if lr_pred is not None else None

    # Sort: Critical first, then High, Medium, Low
    severity_order = {"Critical": 0, "High": 1, "Medium": 2, "Low": 3}
    items.sort(key=lambda i: severity_order.get(i.severity, 9))

    return ProgramGapReport(
        program_id=program.id,
        program_name=program.name,
        university=program.university,
        prereq_match_score=pmatch.match_score,
        gpa_gap=round(gpa_gap, 3),
        admission_prob=admission_prob,
        prob_low=prob_low,
        prob_high=prob_high,
        items=items,
    )
