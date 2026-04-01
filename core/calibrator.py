# Copyright (C) 2026 MasterAgentAI. All rights reserved.
# Licensed under AGPL-3.0. See LICENSE for details.
# SPDX-License-Identifier: AGPL-3.0-only
"""Calibration engine — tunes scoring weights using real admission outcomes.

Uses real admission data (accepted/rejected) to:
1. Compute per-program acceptance thresholds
2. Adjust the school ranker's reach/target/safety classification
3. Estimate feature importance for admission decisions
4. Generate accuracy metrics for the current scoring model

The calibration uses logistic-style scoring — no external ML libraries needed.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .admission_data import (
    AdmissionRecord,
    ProgramStats,
    compute_all_program_stats,
    compute_program_stats,
)

# ---------------------------------------------------------------------------
# Calibrated thresholds
# ---------------------------------------------------------------------------


@dataclass
class ProgramThreshold:
    """Calibrated thresholds for a single program."""

    program_id: str = ""

    # GPA thresholds (on 4.0 scale)
    gpa_floor: float = 0.0  # below this -> almost certainly rejected
    gpa_target: float = 0.0  # above this -> competitive
    gpa_safe: float = 0.0  # above this -> strong safety

    # Background tier threshold
    max_bg_tier_accepted: int = 5  # highest tier (worst) still accepted

    # Intern score threshold
    min_intern_score_accepted: float = 0.0

    # Observed rates
    observed_acceptance_rate: float = 0.0

    # Confidence (based on sample size)
    sample_size: int = 0
    confidence: str = "low"  # low / medium / high

    # Feature weights for this program
    feature_weights: dict[str, float] = field(default_factory=dict)


@dataclass
class CalibrationResult:
    """Output of the full calibration process."""

    program_thresholds: dict[str, ProgramThreshold] = field(default_factory=dict)
    global_feature_weights: dict[str, float] = field(default_factory=dict)
    accuracy_report: dict[str, Any] = field(default_factory=dict)
    recommendations: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Core calibration
# ---------------------------------------------------------------------------


def _confidence_level(n: int) -> str:
    """Determine confidence based on sample size."""
    if n >= 30:
        return "high"
    if n >= 10:
        return "medium"
    return "low"


def calibrate_program(
    stats: ProgramStats,
    records: list[AdmissionRecord],
) -> ProgramThreshold:
    """Calibrate thresholds for a single program based on real data.

    Parameters
    ----------
    stats:
        Pre-computed program statistics.
    records:
        All records (filtered internally to this program).

    Returns
    -------
    ProgramThreshold
        Data-driven thresholds for classification decisions.
    """
    prog_records = [r for r in records if r.program == stats.program_id]
    accepted = [r for r in prog_records if r.result == "accepted"]

    threshold = ProgramThreshold(
        program_id=stats.program_id,
        sample_size=stats.total_records,
        confidence=_confidence_level(stats.total_records),
        observed_acceptance_rate=stats.observed_acceptance_rate,
    )

    if accepted:
        # GPA floor: minimum GPA among accepted applicants
        gpas_accepted = [r.gpa_normalized for r in accepted]
        threshold.gpa_floor = min(gpas_accepted)
        threshold.gpa_target = sum(gpas_accepted) / len(gpas_accepted)
        # Safe threshold: 90th percentile of accepted
        sorted_gpas = sorted(gpas_accepted)
        p90_idx = int(len(sorted_gpas) * 0.9)
        threshold.gpa_safe = sorted_gpas[min(p90_idx, len(sorted_gpas) - 1)]

        # Background tier
        threshold.max_bg_tier_accepted = max(r.bg_tier for r in accepted)

        # Intern score
        intern_scores = [r.intern_score for r in accepted]
        threshold.min_intern_score_accepted = min(intern_scores)

    if stats.feature_importance:
        threshold.feature_weights = stats.feature_importance

    return threshold


def calibrate_all(
    records: list[AdmissionRecord],
) -> CalibrationResult:
    """Run full calibration across all programs in the dataset.

    Parameters
    ----------
    records:
        All admission records loaded from CSV.

    Returns
    -------
    CalibrationResult
        Thresholds, global weights, accuracy metrics, and recommendations.
    """
    all_stats = compute_all_program_stats(records)

    program_thresholds: dict[str, ProgramThreshold] = {}
    for pid, stats in all_stats.items():
        program_thresholds[pid] = calibrate_program(stats, records)

    # Global feature weights (average across programs with enough data)
    global_weights = _compute_global_weights(program_thresholds)

    # Accuracy via leave-one-out cross-validation (no data leakage)
    accuracy = _evaluate_accuracy_cv(records)

    # Recommendations
    recommendations = _generate_recommendations(
        program_thresholds, all_stats, accuracy
    )

    return CalibrationResult(
        program_thresholds=program_thresholds,
        global_feature_weights=global_weights,
        accuracy_report=accuracy,
        recommendations=recommendations,
    )


def _compute_global_weights(
    thresholds: dict[str, ProgramThreshold],
) -> dict[str, float]:
    """Average feature weights across programs with sufficient data."""
    feature_sums: dict[str, float] = {}
    feature_counts: dict[str, int] = {}

    for pt in thresholds.values():
        if pt.confidence in ("medium", "high") and pt.feature_weights:
            for feat, weight in pt.feature_weights.items():
                feature_sums[feat] = feature_sums.get(feat, 0.0) + abs(weight)
                feature_counts[feat] = feature_counts.get(feat, 0) + 1

    if not feature_sums:
        return {}

    # Normalize to sum to 1
    raw = {
        feat: feature_sums[feat] / feature_counts[feat]
        for feat in feature_sums
    }
    total = sum(raw.values()) or 1.0
    return {feat: round(val / total, 3) for feat, val in sorted(raw.items(), key=lambda x: -x[1])}


# ---------------------------------------------------------------------------
# Accuracy evaluation
# ---------------------------------------------------------------------------


_DEFAULT_WEIGHTS: dict[str, float] = {
    "gpa": 0.35,
    "bg_tier": 0.20,
    "intern": 0.20,
    "research": 0.10,
    "paper": 0.05,
    "gender_f": 0.05,
    "domestic": 0.05,
}


def _get_prediction_weights(
    threshold: ProgramThreshold,
) -> dict[str, float]:
    """Return feature weights — learned if available, else defaults.

    If the program has calibrated feature_weights (from effect size
    analysis), normalize them to sum to 1.0 and use those.  Otherwise
    fall back to ``_DEFAULT_WEIGHTS``.
    """
    fw = threshold.feature_weights
    if fw and threshold.confidence in ("medium", "high"):
        raw = {k: abs(v) for k, v in fw.items()}
        total = sum(raw.values()) or 1.0
        return {k: v / total for k, v in raw.items()}
    return dict(_DEFAULT_WEIGHTS)


def predict_outcome(
    record: AdmissionRecord,
    threshold: ProgramThreshold,
) -> str:
    """Predict admission outcome based on calibrated thresholds.

    Uses learned feature weights when available (medium/high
    confidence), otherwise falls back to default weights.

    Returns 'accepted', 'rejected', or 'borderline'.
    """
    weights = _get_prediction_weights(threshold)
    score = 0.0
    max_score = 0.0

    # GPA component
    w = weights.get("gpa", 0.35)
    max_score += w
    if threshold.gpa_target > 0:
        gpa_ratio = record.gpa_normalized / threshold.gpa_target
        score += w * min(1.0, gpa_ratio)

    # Background tier (lower tier = better)
    w = weights.get("bg_tier", 0.20)
    max_score += w
    bg_ratio = 1.0 - (record.bg_tier - 1) / 4.0
    score += w * max(0.0, bg_ratio)

    # Intern score
    w = weights.get("intern", 0.20)
    max_score += w
    if record.intern_score > 0:
        score += w * min(1.0, record.intern_score / 8.0)

    # Research bonus
    w = weights.get("research", 0.10)
    max_score += w
    if record.has_research:
        score += w * 1.0

    # Paper bonus
    w = weights.get("paper", 0.05)
    max_score += w
    if record.has_paper:
        score += w * 1.0

    # Gender (small factor)
    w = weights.get("gender_f", 0.05)
    max_score += w
    if record.gender == "F":
        score += w * 1.0
    elif record.gender == "M":
        score += w * 0.5

    # Domestic advantage (small factor)
    w = weights.get("domestic", 0.05)
    max_score += w
    nat = record.nationality_canonical
    if nat == "domestic":
        score += w * 1.0
    elif nat == "hk_tw":
        score += w * 0.7
    elif nat == "china":
        score += w * 0.5
    else:
        score += w * 0.6

    # Classify based on score ratio
    ratio = score / max_score if max_score > 0 else 0.0

    # Adjust threshold by program selectivity
    accept_threshold = 0.55
    if threshold.observed_acceptance_rate < 0.15:
        accept_threshold = 0.65
    elif threshold.observed_acceptance_rate > 0.50:
        accept_threshold = 0.45

    if ratio >= accept_threshold:
        return "accepted"
    if ratio >= accept_threshold - 0.15:
        return "borderline"
    return "rejected"


def _evaluate_accuracy_cv(
    records: list[AdmissionRecord],
) -> dict[str, Any]:
    """Evaluate accuracy via leave-one-out cross-validation.

    For each record, calibrate on all OTHER records, then predict
    the held-out record.  This prevents train/test data leakage.
    """
    results: dict[str, Any] = {
        "total_predictions": 0,
        "correct": 0,
        "incorrect": 0,
        "borderline": 0,
        "per_program": {},
        "method": "leave-one-out CV",
    }

    decidable = [r for r in records if r.result in ("accepted", "rejected")]

    for i, held_out in enumerate(decidable):
        # Train on everything except the held-out record.
        train = [r for j, r in enumerate(decidable) if j != i]

        # Need at least 1 accepted + 1 rejected for this program.
        prog_train = [r for r in train if r.program == held_out.program]
        has_acc = any(r.result == "accepted" for r in prog_train)
        has_rej = any(r.result == "rejected" for r in prog_train)
        if not has_acc or not prog_train:
            continue

        stats = compute_program_stats(train, held_out.program)
        threshold = calibrate_program(stats, train)

        # Use learned weights only if there's enough contrast.
        if not has_rej:
            threshold.feature_weights = {}

        predicted = predict_outcome(held_out, threshold)
        results["total_predictions"] += 1

        if predicted == "borderline":
            results["borderline"] += 1
        elif predicted == held_out.result:
            results["correct"] += 1
        else:
            results["incorrect"] += 1

        pid = held_out.program
        if pid not in results["per_program"]:
            results["per_program"][pid] = {
                "correct": 0, "incorrect": 0,
                "borderline": 0, "total": 0,
            }
        pstats = results["per_program"][pid]
        pstats["total"] += 1
        if predicted == "borderline":
            pstats["borderline"] += 1
        elif predicted == held_out.result:
            pstats["correct"] += 1
        else:
            pstats["incorrect"] += 1

    decided = results["correct"] + results["incorrect"]
    results["accuracy"] = (
        results["correct"] / decided if decided > 0 else 0.0
    )
    return results


# ---------------------------------------------------------------------------
# Recommendations
# ---------------------------------------------------------------------------


def _generate_recommendations(
    thresholds: dict[str, ProgramThreshold],
    all_stats: dict[str, ProgramStats],
    accuracy: dict[str, Any],
) -> list[str]:
    """Generate actionable recommendations from calibration results."""
    recs: list[str] = []

    # Data quantity check
    total = sum(t.sample_size for t in thresholds.values())
    low_data = [pid for pid, t in thresholds.items() if t.confidence == "low"]

    if total < 50:
        recs.append(
            f"Dataset has only {total} records. Collect more data for "
            "reliable calibration (target: 100+ records, 30+ per program)."
        )

    if low_data:
        recs.append(
            f"Low confidence for {len(low_data)} programs: "
            f"{', '.join(low_data[:5])}. Need 10+ records per program."
        )

    # Accuracy check
    acc = accuracy.get("accuracy", 0)
    if acc < 0.6:
        recs.append(
            f"Current model accuracy is {acc:.0%}. Consider adding "
            "more features (coursework detail, recommendation quality)."
        )
    elif acc >= 0.8:
        recs.append(
            f"Model accuracy is {acc:.0%} — strong predictive power. "
            "Continue collecting data to maintain reliability."
        )

    # Feature insights
    for pid, threshold in thresholds.items():
        if threshold.feature_weights:
            fw = threshold.feature_weights
            top_feat = max(fw, key=lambda k: abs(fw[k]))
            if abs(threshold.feature_weights[top_feat]) > 1.0:
                recs.append(
                    f"{pid}: '{top_feat}' is the strongest differentiator "
                    f"(effect size: {threshold.feature_weights[top_feat]:.2f})."
                )

    return recs


# ---------------------------------------------------------------------------
# Integration: generate school_ranker overrides
# ---------------------------------------------------------------------------


def generate_ranker_overrides(
    calibration: CalibrationResult,
) -> dict[str, dict[str, Any]]:
    """Generate per-program overrides for the school ranker.

    These overrides can replace the hardcoded thresholds in
    ``school_ranker._classify()`` with data-driven values.

    Returns
    -------
    dict[str, dict]
        Mapping of program_id to override dict with keys:
        ``reach_gpa_threshold``, ``safety_gpa_threshold``,
        ``observed_acceptance_rate``.
    """
    overrides: dict[str, dict[str, Any]] = {}

    for pid, threshold in calibration.program_thresholds.items():
        if threshold.confidence == "low" and threshold.sample_size < 5:
            continue

        overrides[pid] = {
            "reach_gpa_threshold": round(threshold.gpa_target, 2),
            "safety_gpa_threshold": round(threshold.gpa_safe, 2),
            "observed_acceptance_rate": round(threshold.observed_acceptance_rate, 3),
            "gpa_floor": round(threshold.gpa_floor, 2),
            "confidence": threshold.confidence,
            "sample_size": threshold.sample_size,
        }

    return overrides
