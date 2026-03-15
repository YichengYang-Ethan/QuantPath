"""School ranking and classification engine.

Classifies programmes into reach / target / safety buckets and computes
a per-programme fit score (0-100) based on:
    - User GPA vs programme average GPA
    - Prerequisite match score
    - Programme acceptance rate
    - Overall evaluation score from the profile evaluator

When calibration data is available, data-driven thresholds override the
default heuristic rules.
"""

from __future__ import annotations

from typing import Any, Optional

from .models import EvaluationResult, ProgramData, UserProfile
from .prerequisite_matcher import match_prerequisites

# ===================================================================
# Classification logic
# ===================================================================


def _classify(
    user_gpa: float,
    program_avg_gpa: float,
    acceptance_rate: float,
    overrides: Optional[dict[str, Any]] = None,
) -> str:
    """Classify a programme as reach, target, or safety.

    When *overrides* are provided (from calibration), uses data-driven
    GPA thresholds instead of the default heuristic rules.

    Default rules (applied in order):
        1. If acceptance_rate < 0.08 OR user_gpa < program_avg_gpa:
           ``"reach"``
        2. If acceptance_rate > 0.15 AND user_gpa >= program_avg_gpa + 0.1:
           ``"safety"``
        3. Otherwise: ``"target"``

    Calibrated rules (when overrides provided):
        1. If user_gpa < gpa_floor: ``"reach"``
        2. If user_gpa >= safety_gpa_threshold: ``"safety"``
        3. If user_gpa >= reach_gpa_threshold: ``"target"``
        4. Otherwise: ``"reach"``
    """
    # Use calibrated thresholds when available
    if overrides and overrides.get("confidence") in ("medium", "high"):
        gpa_floor = overrides.get("gpa_floor", 0)
        reach_threshold = overrides.get("reach_gpa_threshold", program_avg_gpa)
        safety_threshold = overrides.get("safety_gpa_threshold", program_avg_gpa + 0.1)

        if user_gpa < gpa_floor:
            return "reach"
        if user_gpa >= safety_threshold:
            return "safety"
        if user_gpa >= reach_threshold:
            return "target"
        return "reach"

    # Fallback: default heuristic rules
    acceptance_rate = acceptance_rate or 0.15
    program_avg_gpa = program_avg_gpa or 3.80
    if acceptance_rate < 0.08 or user_gpa < program_avg_gpa:
        return "reach"
    if acceptance_rate > 0.15 and user_gpa >= program_avg_gpa + 0.1:
        return "safety"
    return "target"


def _compute_fit_score(
    user_gpa: float,
    program_avg_gpa: float,
    acceptance_rate: float,
    prereq_match_score: float,
    overall_eval_score: float,
) -> float:
    """Compute a composite fit score (0-100) for a programme.

    Components (out of 100):
        GPA closeness           25 pts
        Prerequisite match      30 pts
        Acceptance feasibility  20 pts
        Academic profile        25 pts

    Parameters
    ----------
    user_gpa:
        Applicant's cumulative GPA (0-4.0 scale).
    program_avg_gpa:
        Programme's reported average GPA of admits.
    acceptance_rate:
        Programme's acceptance rate (0-1).
    prereq_match_score:
        Fraction of required prerequisites met (0-1).
    overall_eval_score:
        Overall score from the profile evaluator (0-10).
    """
    # Handle None values
    program_avg_gpa = program_avg_gpa or 3.80
    acceptance_rate = acceptance_rate or 0.15

    # GPA closeness: 25 points.
    # Full marks if user GPA >= program avg; scales down as gap increases.
    gpa_diff = user_gpa - program_avg_gpa
    if gpa_diff >= 0:
        gpa_pts = 25.0
    else:
        # Lose ~6 pts per 0.1 GPA below average, floored at 0.
        gpa_pts = max(0.0, 25.0 + gpa_diff * 60)

    # Prerequisite match: 30 points (linear).
    prereq_pts = prereq_match_score * 30.0

    # Acceptance feasibility: 20 points.
    # Higher acceptance rate -> easier -> more points.
    # 0.20+ -> full marks, 0.05 -> 5 pts, linear between.
    if acceptance_rate >= 0.20:
        accept_pts = 20.0
    elif acceptance_rate <= 0.03:
        accept_pts = 2.0
    else:
        accept_pts = 2.0 + (acceptance_rate - 0.03) / (0.20 - 0.03) * 18.0

    # Academic profile: 25 points (eval score mapped from 0-10 to 0-25).
    profile_pts = (overall_eval_score / 10.0) * 25.0

    return round(gpa_pts + prereq_pts + accept_pts + profile_pts, 1)


# ===================================================================
# Public API
# ===================================================================


def rank_schools(
    profile: UserProfile,
    programs: list[ProgramData],
    evaluation: EvaluationResult,
    calibration_overrides: Optional[dict[str, dict[str, Any]]] = None,
) -> dict[str, Any]:
    """Rank and classify a set of programmes for the given applicant.

    Parameters
    ----------
    profile:
        The applicant's profile (needed for prerequisite matching).
    programs:
        List of programme data objects to evaluate.
    evaluation:
        A pre-computed :class:`EvaluationResult` from the profile
        evaluator.
    calibration_overrides:
        Optional dict of per-program overrides from the calibrator.
        When provided, classification uses data-driven thresholds.

    Returns
    -------
    dict
        Structure::

            {
                "reach":  [sorted list of programme dicts],
                "target": [sorted list of programme dicts],
                "safety": [sorted list of programme dicts],
                "all":    [all programmes sorted by fit_score desc],
            }

        Each programme dict contains: ``program_id``, ``name``,
        ``university``, ``category`` (reach/target/safety),
        ``fit_score``, ``prereq_match_score``, ``acceptance_rate``,
        ``avg_gpa``.
    """
    results: list[dict[str, Any]] = []
    overrides = calibration_overrides or {}

    for prog in programs:
        # Prerequisite matching.
        pmatch = match_prerequisites(profile, prog)

        # Classification (with optional data-driven overrides).
        prog_overrides = overrides.get(prog.id)
        category = _classify(
            user_gpa=profile.gpa,
            program_avg_gpa=prog.avg_gpa,
            acceptance_rate=prog.acceptance_rate,
            overrides=prog_overrides,
        )

        # Fit score.
        fit = _compute_fit_score(
            user_gpa=profile.gpa,
            program_avg_gpa=prog.avg_gpa,
            acceptance_rate=prog.acceptance_rate,
            prereq_match_score=pmatch.match_score,
            overall_eval_score=evaluation.overall_score,
        )

        result_entry: dict[str, Any] = {
            "program_id": prog.id,
            "name": prog.name,
            "university": prog.university,
            "category": category,
            "fit_score": fit,
            "prereq_match_score": pmatch.match_score,
            "acceptance_rate": prog.acceptance_rate,
            "avg_gpa": prog.avg_gpa,
        }

        # Add calibration info if available
        if prog_overrides:
            result_entry["calibrated"] = True
            result_entry["confidence"] = prog_overrides.get("confidence", "low")
            result_entry["sample_size"] = prog_overrides.get("sample_size", 0)

        results.append(result_entry)

    # Sort each bucket by fit_score descending.
    results.sort(key=lambda r: -r["fit_score"])

    output: dict[str, Any] = {
        "reach": [r for r in results if r["category"] == "reach"],
        "target": [r for r in results if r["category"] == "target"],
        "safety": [r for r in results if r["category"] == "safety"],
        "all": results,
    }

    return output
