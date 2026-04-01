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

from .lr_predictor import predict_prob_full, predict_prob_v2
from .models import EvaluationResult, ProgramData, UserProfile
from .prerequisite_matcher import match_prerequisites

# Probability thresholds for data-driven classification
_PROB_SAFETY = 0.70   # P(admit) >= 70% → safety
_PROB_REACH  = 0.40   # P(admit) < 40%  → reach
# 40% <= P < 70% → target

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
    admission_prob: Optional[float] = None,
) -> float:
    """Compute a composite fit score (0-100) for a programme.

    Components (out of 100):
        GPA closeness           25 pts
        Prerequisite match      30 pts
        Acceptance feasibility  20 pts  (LR P(admit) when available)
        Academic profile        25 pts

    Parameters
    ----------
    user_gpa:
        Applicant's cumulative GPA (0-4.0 scale).
    program_avg_gpa:
        Programme's reported average GPA of admits.
    acceptance_rate:
        Programme's acceptance rate (0-1). Used as fallback when no LR model.
    prereq_match_score:
        Fraction of required prerequisites met (0-1).
    overall_eval_score:
        Overall score from the profile evaluator (0-10).
    admission_prob:
        Bias-corrected P(admit) from LR model (0-1). When provided, replaces
        the acceptance_rate heuristic for the acceptance feasibility component,
        making the score profile-specific rather than program-average.
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
    # When LR model is available, use profile-specific P(admit).
    # Otherwise fall back to the program-level acceptance rate heuristic.
    if admission_prob is not None:
        accept_pts = admission_prob * 20.0
    elif acceptance_rate >= 0.20:
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
    projected: bool = False,
    **kwargs,
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

    # GRE Quant for LR prediction
    gre_quant = profile.test_scores.gre_quant

    for prog in programs:
        # Prerequisite matching.
        pmatch = match_prerequisites(profile, prog)

        # --- Classification: LR probability first, heuristic fallback ---
        # Use full prediction (bias-corrected + profile-aware + CI)
        # Use v2 (GPBoost) when use_v2=True, otherwise v1 (per-program LR)
        if kwargs.get("use_v2"):
            lr_pred = predict_prob_v2(prog.id, profile.gpa, gre_quant, profile)
        else:
            lr_pred = predict_prob_full(prog.id, profile.gpa, gre_quant, profile)
        admission_prob = lr_pred.prob if lr_pred is not None else None

        if admission_prob is not None:
            # Data-driven: use LR probability thresholds
            if admission_prob >= _PROB_SAFETY:
                category = "safety"
            elif admission_prob >= _PROB_REACH:
                category = "target"
            else:
                category = "reach"
        else:
            # No LR model — fall back to heuristic classification
            prog_overrides = overrides.get(prog.id)
            category = _classify(
                user_gpa=profile.gpa,
                program_avg_gpa=prog.avg_gpa,
                acceptance_rate=prog.acceptance_rate,
                overrides=prog_overrides,
            )

        # Fit score — profile-specific when LR prob available
        eval_score = evaluation.overall_score
        fit = _compute_fit_score(
            user_gpa=profile.gpa,
            program_avg_gpa=prog.avg_gpa,
            acceptance_rate=prog.acceptance_rate,
            prereq_match_score=pmatch.match_score,
            overall_eval_score=eval_score,
            admission_prob=admission_prob,
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
            "admission_prob": admission_prob,
            "prob_low": lr_pred.prob_low if lr_pred is not None else None,
            "prob_high": lr_pred.prob_high if lr_pred is not None else None,
        }

        # Add calibration info if available
        prog_overrides = overrides.get(prog.id)
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
